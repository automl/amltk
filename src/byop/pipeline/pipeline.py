"""The pipeline class used to represent a pipeline of steps.

This module exposes a Pipelne class that wraps a chain of `Component`, `Split`
and `Choice` components, created through the `step`, `choice` and `split`
api functions from `byop.pipeline`.
"""
from __future__ import annotations

from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    overload,
)
from uuid import uuid4

from attrs import frozen
from more_itertools import duplicates_everseen, first_true

from byop.pipeline.step import Step
from byop.typing import Config, Key, Name, Seed, Space

T = TypeVar("T")  # Dummy typevar

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from byop.configuring import Configurer
    from byop.parsing import SpaceParser
    from byop.pipeline.components import Split


@frozen(kw_only=True)
class Pipeline(Generic[Key, Name]):
    """Base class implementing search routines over steps.

    Attributes:
        name: The name of the pipeline
        steps: The steps in the pipeline
    """

    name: Name
    steps: list[Step[Key]]

    @property
    def head(self) -> Step[Key]:
        """The first step in the pipeline."""
        return self.steps[0]

    @property
    def tail(self) -> Step[Key]:
        """The last step in the pipeline."""
        return self.steps[-1]

    def __contains__(self, key: Key | Step[Key]) -> bool:
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

    def __iter__(self) -> Iterator[Step[Key]]:
        return self.steps.__iter__()

    def __or__(self, other: Step[Key] | Pipeline[Key, Name]) -> Pipeline[Key, Name]:
        """Append a step or pipeline to this one and return a new one."""
        return self.append(other)

    def iter(self) -> Iterator[Step[Key]]:
        """Iterate over the top layer of the pipeline.

        Yields:
            Step[Key]
        """
        yield from iter(self.steps)

    def traverse(self) -> Iterator[Step[Key]]:
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
        self, key: Key | Callable[[Step[Key]], bool], default: T, *, deep: bool = ...
    ) -> Step[Key] | T:
        ...

    @overload
    def find(
        self, key: Key | Callable[[Step[Key]], bool], *, deep: bool = ...
    ) -> Step[Key] | None:
        ...

    def find(
        self,
        key: Key | Callable[[Step[Key]], bool],
        default: T | None = None,
        *,
        deep: bool = True,
    ) -> Step[Key] | T | None:
        """Find a step in the pipeline.

        Args:
            key: The key to search for or a function that returns True if the step
                is the desired step
            default (optional):
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
        choices: Mapping[Key, Key],
        *,
        name: Name | None = None,
    ) -> Pipeline[Key, Name]:
        """Select particular choices from the pipeline.

        Args:
            choices: A mapping of the choice name to the choice to select
            name: A name to give to the new pipeline returned. Defaults to the current

        Returns:
            A new pipeline with the selected choices
        """
        return Pipeline.create(
            self.head.select(choices),
            name=self.name if name is None else name,
        )

    def remove(
        self,
        step: Key | list[Key],
        *,
        name: Name | None = None,
    ) -> Pipeline[Key, Name]:
        """Remove a step from the pipeline.

        Args:
            step: The name of the step(s) to remove
            name (optional): A name to give to the new pipeline returned. Defaults to
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

    def append(
        self,
        nxt: Pipeline[Key, Name] | Step[Key],
        *,
        name: Name | None = None,
    ) -> Pipeline[Key, Name]:
        """Append a step or pipeline to this one and return a new one.

        Args:
            nxt: The step or pipeline to append
            name (optional): A name to give to the new pipeline returned. Defaults to
                the current pipelines name

        Returns:
            Pipeline: A new pipeline with the step appended
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
        key: Key | dict[Key, Step[Key]],
        step: Step[Key] | None = None,
        *,
        name: Name | None = None,
    ) -> Pipeline[Key, Name]:
        """Replace a step in the pipeline.

        Args:
            key: The key of the step to replace or a dictionary of steps to replace
            step (optional): The step to replace the old step with. Only used if key is
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
        parser: Callable[[Pipeline], Space] | Callable[[Pipeline, Seed | None], Space],
        *,
        seed: Seed | None = ...,
    ) -> Space:
        ...

    @overload
    def space(
        self,
        parser: SpaceParser[Space],
        *,
        seed: Seed | None = ...,
    ) -> Space:
        ...

    def space(
        self,
        parser: (
            Literal["auto"]
            | Literal["configspace"]
            | type[ConfigurationSpace]
            | Callable[[Pipeline], Space]
            | Callable[[Pipeline, Seed | None], Space]
            | SpaceParser[Space]
        ) = "auto",
        *,
        seed: Seed | None = None,
    ) -> Space | ConfigurationSpace | Any:
        """Get the space for the pipeline.

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
            seed (optional): The seed to seed the space with if applicable.

        Returns:
            The space for the pipeline
        """
        from byop.parsing import parse  # Prevent circular imports

        return parse(self, parser=parser, seed=seed)

    def configure(
        self,
        config: Config,
        *,
        configurer: (
            Literal["auto"]
            | Configurer
            | Callable[[Pipeline[Key, Name], Config], Pipeline[Key, Name]]
        ) = "auto",
        rename: bool | Name = False,
    ) -> Pipeline[Key, Name]:
        """Configure the pipeline with the given configuration.

        This takes a pipeline with spaces and choices and trims it down based on the
        configuration. For example, choosing selected steps and setting the `config`
        of steps with those present in the `config` object given to this function.

        Args:
            config: The configuration to use
            configurer: The configurer to use. Default is `"auto"`.
                * If `"auto"` is provided, the assembler will attempt to automatically
                    figure out the kind of Configurer to use from the config.
                * If `configurer` is a configurer type, we will attempt to use that.
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

    @classmethod
    @overload
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
    ) -> Pipeline[Key, str]:
        ...

    @classmethod
    @overload
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
        name: Name,
    ) -> Pipeline[Key, Name]:
        ...

    @classmethod
    def create(
        cls,
        *steps: Step[Key] | Pipeline[Key, Name] | Iterable[Step[Key]],
        name: Name | None = None,
    ) -> Pipeline[Key, Name] | Pipeline[Key, str]:
        """Create a pipeline from a sequence of steps.

        Args:
            *steps: The steps to create the pipeline from
            name (optional): The name of the pipeline. Defaults to a uuid

        Returns:
            Pipeline
        """
        # Expand out any pipelines in the init
        expanded = [s.steps if isinstance(s, Pipeline) else s for s in steps]
        step_sequence = list(Step.chain(*expanded))

        if name is not None:
            return cls(name=name, steps=step_sequence)

        return Pipeline(name=str(uuid4()), steps=step_sequence)
