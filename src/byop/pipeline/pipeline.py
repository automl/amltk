"""The pipeline class used to represent a pipeline of steps.

This module exposes a Pipelne class that wraps a chain of `Component`, `Split`
and `Choice` components, created through the `step`, `choice` and `split`
api functions from `byop.pipeline`.
"""
from __future__ import annotations

import logging
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)
from uuid import uuid4

from attrs import field, frozen
from more_itertools import duplicates_everseen, first_true

from byop.pipeline.components import Searchable
from byop.pipeline.step import Step
from byop.types import Config, Seed, Space

T = TypeVar("T")  # Dummy typevar
B = TypeVar("B")  # Built pipeline

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from byop.optuna.space import OptunaSearchSpace
    from byop.pipeline.components import Split
    from byop.samplers import Sampler

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class Pipeline:
    """Base class implementing search routines over steps.

    Attributes:
        name: The name of the pipeline
        steps: The steps in the pipeline
        modules: Additional modules to associate with the pipeline
    """

    name: str
    steps: list[Step]
    modules: Mapping[str, Pipeline] = field(factory=dict)
    searchables: Mapping[str, Searchable] = field(factory=dict)

    @property
    def head(self) -> Step:
        """The first step in the pipeline."""
        return self.steps[0]

    @property
    def tail(self) -> Step:
        """The last step in the pipeline."""
        return self.steps[-1]

    def __contains__(self, key: str | Step) -> bool:
        """Check if a step is in the pipeline.

        Args:
            key: The name of the step or the step itself

        Returns:
            bool: True if the step is in the pipeline, False otherwise
        """
        key = key.name if isinstance(key, Step) else key
        return self.find(key, deep=True) is not None

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return self.steps.__iter__()

    def __or__(self, other: Step | Pipeline) -> Pipeline:
        """Append a step or pipeline to this one and return a new one."""
        return self.append(other)

    def iter(self) -> Iterator[Step]:
        """Iterate over the top layer of the pipeline.

        Yields:
            Step[Key]
        """
        yield from iter(self.steps)

    def traverse(self) -> Iterator[Step]:
        """Traverse the pipeline in a depth-first manner.

        Yields:
            Step[Key]
        """
        yield from chain.from_iterable(step.traverse() for step in self.steps)

    def walk(self) -> Iterator[tuple[list[Split], list[Step], Step]]:
        """Walk the pipeline in a depth-first manner.

        This is similar to traverse, but yields the splits that lead to the step along
        with any parents in a chain with that step (which does not include the splits)

        Yields:
            (splits, parents, step)
        """
        yield from self.head.walk(splits=[], parents=[])

    @overload
    def find(
        self,
        key: str | Callable[[Step], bool],
        default: T,
        *,
        deep: bool = ...,
    ) -> Step | T:
        ...

    @overload
    def find(
        self, key: str | Callable[[Step], bool], *, deep: bool = ...
    ) -> Step | None:
        ...

    def find(
        self,
        key: str | Callable[[Step], bool],
        default: T | None = None,
        *,
        deep: bool = True,
    ) -> Step | T | None:
        """Find a step in the pipeline.

        Args:
            key: The key to search for or a function that returns True if the step
                is the desired step
            default:
                The value to return if the step is not found. Defaults to None
            deep:
                Whether to search the entire pipeline or just the top layer.

        Returns:
            The step if found, otherwise the default value. Defaults to None
        """
        result = first_true(
            iterable=self.traverse() if deep else self.iter(),
            pred=key if callable(key) else (lambda step: step.name == key),
        )
        return result if result is not None else default

    def select(
        self,
        choices: Mapping[str, str],
        *,
        name: str | None = None,
    ) -> Pipeline:
        """Select particular choices from the pipeline.

        Args:
            choices: A mapping of the choice name to the choice to select
            name: A name to give to the new pipeline returned. Defaults to the current

        Returns:
            A new pipeline with the selected choices
        """
        return self.create(
            self.head.select(choices),
            name=self.name if name is None else name,
        )

    def remove(self, step: str | list[str], *, name: str | None = None) -> Pipeline:
        """Remove a step from the pipeline.

        Args:
            step: The name of the step(s) to remove
            name: A name to give to the new pipeline returned. Defaults to
                the current pipelines name

        Returns:
            A new pipeline with the step removed
        """
        # NOTE: We explicitly use a list instead of a Sequence for multiple steps.
        # This is because technically you could have a single Key = tuple(X, Y, Z),
        # which is a Sequence.
        # This problem also arises more simply in the case where Key = str.
        # Hence, by explicitly checking for the concrete type, `list`, we can
        # avoid this problem.
        return self.create(
            self.head.remove(step if isinstance(step, list) else [step]),
            name=name if name is not None else self.name,
        )

    def append(self, nxt: Pipeline | Step, *, name: str | None = None) -> Pipeline:
        """Append a step or pipeline to this one and return a new one.

        Args:
            nxt: The step or pipeline to append
            name: A name to give to the new pipeline returned. Defaults to
                the current pipelines name

        Returns:
            A new pipeline with the step appended
        """
        if isinstance(nxt, Pipeline):
            nxt = nxt.head

        return self.create(
            self.steps,
            nxt.iter(),
            name=name if name is not None else self.name,
        )

    def replace(
        self,
        key: str | dict[str, Step],
        step: Step | None = None,
        *,
        name: str | None = None,
    ) -> Pipeline:
        """Replace a step in the pipeline.

        Args:
            key: The key of the step to replace or a dictionary of steps to replace
            step: The step to replace the old step with. Only used if key is
                a single key
            name: A name to give to the new pipeline returned. Defaults to the current

        Returns:
            A new pipeline with the step replaced
        """
        if isinstance(key, dict) and step is not None:
            raise ValueError("Cannot specify both dictionary of keys and step")

        if not isinstance(key, dict):
            if step is None:
                raise ValueError("Must specify step to replace with")
            replacements = {key: step}
        else:
            replacements = key

        return self.create(
            self.head.replace(replacements),
            name=self.name if name is None else name,
        )

    def validate(self) -> None:
        """Validate the pipeline for any invariants.

        Intended for use as an opt-in during development

        Raises:
            AssertionError: If a duplicate name is found for a step in the pipeline
        """
        # Check that we do not have any keys with the same Hash
        dupe_steps = list(duplicates_everseen(self.traverse()))
        assert not any(dupe_steps), f"Duplicates in pipeline {dupe_steps}"

    def attach(
        self,
        *,
        modules: Pipeline | Iterable[Pipeline | Step] | None = None,
        searchables: Searchable | Sequence[Searchable] | None = None,
    ) -> Pipeline:
        """Attach modules to the pipeline.

        Args:
            modules: The modules to attach
            searchables: The searchables to attach
        """
        if modules is None:
            modules = []

        if isinstance(modules, Pipeline):
            modules = [modules]

        if searchables is None:
            searchables = []

        if isinstance(searchables, Searchable):
            searchables = [searchables]

        return self.create(
            self.head,
            modules=[*self.modules.values(), *modules],
            searchables=[*self.searchables.values(), *searchables],
            name=self.name,
        )

    def configured(self) -> bool:
        """Whether the pipeline has been configured.

        Returns:
            True if the pipeline has been configured, False otherwise
        """
        return (
            all(step.configured() for step in self.traverse())
            and all(module.configured() for module in self.modules.values())
            and all(searchable.configured() for searchable in self.searchables.values())
        )

    @overload
    def space(
        self, parser: Literal["auto"] = "auto", *, seed: Seed | None = ...
    ) -> Any:
        ...

    @overload
    def space(
        self,
        parser: Literal["configspace"] | type[ConfigurationSpace],
        *,
        seed: Seed | None = ...,
    ) -> ConfigurationSpace:
        ...

    @overload
    def space(
        self,
        parser: Literal["optuna"],
        *,
        seed: Seed | None = ...,
    ) -> OptunaSearchSpace:
        ...

    @overload
    def space(
        self,
        parser: Callable[[Pipeline], Space] | Callable[[Pipeline, Seed | None], Space],
        *,
        seed: Seed | None = ...,
    ) -> Space:
        ...

    def space(
        self,
        parser: (
            Literal["auto"]
            | Literal["configspace"]
            | Literal["optuna"]
            | type[ConfigurationSpace]
            | Callable[[Pipeline], Space]
            | Callable[[Pipeline, Seed | None], Space]
        ) = "auto",
        *,
        seed: Seed | None = None,
    ) -> Space | ConfigurationSpace | OptunaSearchSpace | Any:
        """Get the space for the pipeline.

        If there are any modules or searchables attached to this pipeline,
        these will also be included in the space.

        Args:
            parser: The parser to use for assembling the space. Default is `"auto"`.
                * If `"auto"` is provided, the assembler will attempt to
                automatically figure out the kind of Space to extract from the pipeline.
                * If `"configspace"` is provided, a ConfigurationSpace will be attempted
                to be extracted.
                * If a `type` is provided, it will attempt to infer which parser to use.
                * If `parser` is a parser type, we will attempt to use that.
                * If `parser` is a callable, we will attempt to use that.
                If there are other intuitive ways to indicate the type, please open
                an issue on GitHub and we will consider it!
            seed: The seed to seed the space with if applicable.

        Returns:
            The space for the pipeline
        """
        from byop.parsing import parse  # Prevent circular imports

        return parse(self, parser=parser, seed=seed)

    def sample(
        self,
        space: Space,
        *,
        n: int | None = None,
        sampler: Sampler[Space] | None = None,
        seed: Seed | None = None,
    ) -> Config | list[Config]:
        """Sample a configuration from the space of the pipeline.

        Args:
            space: The space to sample from
            n: The number of configurations to sample. If `None`, a single
                configuration will be sampled. If `n` is greater than 1, a list of
                configurations will be returned.
            sampler: The sampler to use. If `None`, a sampler will be automatically
                chosen based on the type of space that is provided.
            seed: The seed to seed the space with if applicable.

        Returns:
            A configuration sampled from the space of the pipeline
        """
        from byop.samplers import sample

        return sample(space, sampler=sampler, n=n, seed=seed)

    def configure(
        self,
        config: Config,
        *,
        configurer: (Literal["auto"] | Callable[[Pipeline, Config], Pipeline]) = "auto",
        rename: bool | str = False,
    ) -> Pipeline:
        """Configure the pipeline with the given configuration.

        This takes a pipeline with spaces and choices and trims it down based on the
        configuration. For example, choosing selected steps and setting the `config`
        of steps with those present in the `config` object given to this function.

        If there are any modules or searchables attached to this pipeline,
        these will also be configured for you.

        Args:
            config: The configuration to use
            configurer: The configurer to use. Default is `"auto"`.
                * If `"auto"` is provided, the assembler will attempt to automatically
                    figure out how to configure the pipeline.
                * If `configurer` is a callable, we will attempt to use that.
                If there are other intuitive ways to indicate the type, please open an
                issue on GitHub and we will consider it!
            rename: Whether to rename the pipeline. Defaults to `False`.
                * If `True`, the pipeline will be renamed using a random uuid
                * If a Name is provided, the pipeline will be renamed to that name

        Returns:
            A new pipeline with the configuration applied
        """
        from byop.configuring import configure  # Prevent circular imports

        return configure(self, config, configurer=configurer, rename=rename)

    @overload
    def build(self, builder: Literal["auto"] = "auto") -> Any:
        ...

    @overload
    def build(self, builder: Callable[[Pipeline], B]) -> B:
        ...

    def build(
        self,
        builder: (Literal["auto"] | Callable[[Pipeline], B]) = "auto",
    ) -> B | Any:
        """Build the pipeline.

        Args:
            builder: The builder to use. Default is `"auto"`.
                * If `"auto"` is provided, the assembler will attempt to automatically
                    figure out build the pipeline as it can.
                * If `builder` is a callable, we will attempt to use that.

        Returns:
            The built pipeline
        """
        from byop.building import build  # Prevent circular imports

        return build(self, builder=builder)

    def copy(self, *, name: str | None = None) -> Pipeline:
        """Copy the pipeline.

        Returns:
            A copy of the pipeline
        """
        return self.create(self, name=self.name if name is None else name)

    @classmethod
    def create(  # noqa: C901
        cls,
        *steps: Step | Pipeline | Iterable[Step],
        modules: Pipeline | Iterable[Pipeline | Step] | None = None,
        searchables: Searchable | Iterable[Searchable] | None = None,
        name: str | None = None,
    ) -> Pipeline:
        """Create a pipeline from a sequence of steps.

        Args:
            *steps: The steps to create the pipeline from
            name (optional): The name of the pipeline. Defaults to a uuid
            modules (optional): The modules to use for the pipeline
            searchables (optional): The searchables to use for the pipeline

        Returns:
            Pipeline
        """
        # Expand out any pipelines in the init
        expanded = [s.steps if isinstance(s, Pipeline) else s for s in steps]
        step_sequence = list(Step.chain(*expanded))

        if name is None:
            name = str(uuid4())

        if isinstance(modules, Pipeline):
            modules = [modules]

        if isinstance(searchables, Searchable):
            searchables = [searchables]

        # Collect all the modules, turning them into pipelines
        # as required by the internal api
        final_modules: dict[str, Pipeline] = {}
        if modules is not None:
            final_modules = {
                module.name: module.copy()
                if isinstance(module, Pipeline)
                else Pipeline.create(module, name=module.name)
                for module in modules
            }

        # If any of the steps are pipelines and contain modules, attach
        # them to the final modules of this newly created pipeline
        for step in steps:
            if isinstance(step, Pipeline):
                step_modules = {
                    module_name: module.copy()
                    for module_name, module in step.modules.items()
                }

                # If one of the subpipelines has a duplicate module name
                # then we need to raise an error
                duplicates = step_modules.keys() & final_modules.keys()
                if any(duplicates):
                    msg = (
                        "Duplicate module(s) found during pipeline"
                        f" creation {duplicates=}."
                    )
                    raise ValueError(msg)

                final_modules.update(step_modules)

        # Collect all the searchables, copying them over
        final_searchables: dict[str, Searchable] = {}
        if searchables is not None:
            final_searchables = {
                searchable.name: searchable.copy() for searchable in searchables
            }

        # If any of the steps are pipelines, make sure to grab their
        # searchables too, as we will flatten all of these added pipelines
        # to one single pipeline
        for step in steps:
            if isinstance(step, Pipeline):
                step_searchables = {
                    searchable.name: searchable.copy()
                    for searchable in step.searchables.values()
                }
                duplicates = step_searchables.keys() & final_searchables.keys()
                if any(duplicates):
                    msg = (
                        "Duplicate searchable(s) found during pipeline"
                        f" creation {duplicates=}."
                    )
                    raise ValueError(msg)

                final_searchables.update(step_searchables)

        return cls(
            name=name,
            steps=step_sequence,
            modules=final_modules,
            searchables=final_searchables,
        )
