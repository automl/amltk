"""The pipeline class used to represent a pipeline of steps.

This module exposes a Pipeline class that wraps a chain of
[`Component`][amltk.pipeline.Component], [`Split`][amltk.pipeline.Split],
[`Group`][amltk.pipeline.Group] and [`Choice`][amltk.pipeline.Choice]
components, created through the [`step()`][amltk.pipeline.api.step],
[`choice()`][amltk.pipeline.choice], [`split()`][amltk.pipeline.split]
and [`group()`][amltk.pipeline.group] api functions from `amltk.pipeline`.
"""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    overload,
)
from typing_extensions import override
from uuid import uuid4

from attrs import field, frozen

from amltk.functional import classname, mapping_select
from amltk.pipeline.components import Group, Step, prefix_keys
from amltk.richutil import RichRenderable

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.text import TextType

    from amltk.pipeline.parser import Parser
    from amltk.pipeline.sampler import Sampler
    from amltk.types import Config, FidT, Seed, Space

T = TypeVar("T")  # Dummy typevar
B = TypeVar("B")  # Built pipeline

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class Pipeline(RichRenderable):
    """A sequence of steps and operations on them."""

    name: str
    """The name of the pipeline"""

    steps: list[Step]
    """The steps in the pipeline.

    This does not include any steps that are part of a `Split` or `Choice`.
    """

    modules: Mapping[str, Step | Pipeline] = field(factory=dict)
    """Additional modules to associate with the pipeline"""

    meta: Mapping[str, Any] | None = None
    """Additional meta information to associate with the pipeline"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "magenta"

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
        yield from self.head.traverse()

    def walk(self) -> Iterator[tuple[list[Group], list[Step], Step]]:
        """Walk the pipeline in a depth-first manner.

        This is similar to traverse, but yields the groups that lead to the step along
        with any parents in a chain with that step (which does not include the groups)

        Yields:
            (groups, parents, step)
        """
        yield from self.head.walk(groups=[], parents=[])

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
        self,
        key: str | Callable[[Step], bool],
        *,
        deep: bool = ...,
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
        return self.head.find(key, default, deep=deep)

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

    def apply(self, f: Callable[[Step], Step], *, name: str | None = None) -> Pipeline:
        """Apply a function to each step in the pipeline, returning a new pipeline.

        !!! warning "Modifications to pipeline structure"

            Any modifications to pipeline structure will be ignored. This is done by
            providing a `copy()` of the step to the function, rejoining each modified
            step in the pipeline and then returning a new pipeline.

        Args:
            f: The function to apply
            name: A name to give to the new pipeline returned. Defaults to the current

        Returns:
            A new pipeline with the function applied
        """
        return self.create(
            self.head.apply(f),
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

    def attach(
        self,
        *,
        modules: Pipeline | Step | Iterable[Pipeline | Step] | None = None,
    ) -> Pipeline:
        """Attach modules to the pipeline.

        Args:
            modules: The modules to attach
        """
        if modules is None:
            modules = []

        if isinstance(modules, (Step, Pipeline)):
            modules = [modules]

        return self.create(
            self.head,
            modules=[*self.modules.values(), *modules],
            name=self.name,
        )

    def configured(self) -> bool:
        """Whether the pipeline has been configured.

        Returns:
            True if the pipeline has been configured, False otherwise
        """
        return all(step.configured() for step in self.traverse()) and all(
            module.configured() for module in self.modules.values()
        )

    def qualified_name(self, step: str | Step, *, delimiter: str = ":") -> str:
        """Get the qualified name of a substep in the pipeline.

        Args:
            step: The step to get the qualified name of
            delimiter: The delimiter to use between the groups and the step

        Returns:
            The qualified name of the step
        """
        # We use the walk function to get the step along with any groups
        # to get there
        if isinstance(step, Step):
            step = step.name

        found = self.find(step)
        if found is None:
            raise ValueError(f"Step {step} not found in pipeline")

        return found.qualified_name(delimiter=delimiter)

    @overload
    def space(
        self,
        parser: type[Parser[Any, Space]] | Parser[Any, Space],
        *,
        seed: Seed | None = None,
    ) -> Space:
        ...

    @overload
    def space(
        self,
        parser: None = None,
        *,
        seed: Seed | None = None,
    ) -> Any:
        ...

    def space(
        self,
        parser: type[Parser[Any, Space]] | Parser[Any, Space] | None = None,
        *,
        seed: Seed | None = None,
    ) -> Space | Any:
        """Get the space for the pipeline.

        If there are any modules attached to this pipeline,
        these will also be included in the space.

        Args:
            parser: The parser to use for assembling the space. Default is `None`.
                * If `None` is provided, the assembler will attempt to
                automatically figure out the kind of Space to extract from the pipeline.
                * Otherwise we will attempt to use the given Parser.
                If there are other intuitive ways to indicate the type, please open
                an issue on GitHub and we will consider it!
            seed: The seed to use for the space if applicable.

        Raises:
            Parser.Error: If the parser fails to parse the space.

        Returns:
            The space for the pipeline
        """
        from amltk.pipeline.parser import Parser  # Prevent circular imports

        return Parser.try_parse(pipeline_or_step=self, parser=parser, seed=seed)

    def fidelities(self) -> dict[str, FidT]:
        """Get the fidelities for the pipeline.

        Returns:
            The fidelities for the pipeline
        """
        return self.head.fidelities()

    @overload
    def sample(self) -> Config:
        ...

    @overload
    def sample(
        self,
        *,
        n: None = None,
        space: Space | None = ...,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = ...,
        seed: Seed | None = ...,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> Config:
        ...

    @overload
    def sample(
        self,
        *,
        n: int,
        space: Space | None = ...,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = ...,
        seed: Seed | None = ...,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> list[Config]:
        ...

    def sample(
        self,
        *,
        n: int | None = None,
        space: Space | None = None,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = None,
        seed: Seed | None = None,
        duplicates: bool | Iterable[Config] = False,
        max_attempts: int | None = 10,
    ) -> Config | list[Config]:
        """Sample a configuration from the space of the pipeline.

        Args:
            space: The space to sample from. Will be automatically inferred
                if `None` is provided.
            n: The number of configurations to sample. If `None`, a single
                configuration will be sampled. If `n` is greater than 1, a list of
                configurations will be returned.
            sampler: The sampler to use. If `None`, a sampler will be automatically
                chosen based on the type of space that is provided.
            seed: The seed to seed the space with if applicable.
            duplicates: If True, allow duplicate samples. If False, make
                sure all samples are unique. If an Iterable, make sure all
                samples are unique and not in the Iterable.
            max_attempts: The number of times to attempt sampling unique
                configurations before giving up. If `None` will keep
                sampling forever until satisfied.

        Returns:
            A configuration sampled from the space of the pipeline
        """
        from amltk.pipeline.parser import Parser
        from amltk.pipeline.sampler import Sampler

        return Sampler.try_sample(
            space
            if space is not None
            else self.space(parser=sampler if isinstance(sampler, Parser) else None),
            sampler=sampler,
            n=n,
            seed=seed,
            duplicates=duplicates,
            max_attempts=max_attempts,
        )

    def config(self) -> Config:
        """Get the configuration for the pipeline.

        Returns:
            The configuration for the pipeline
        """
        config: dict[str, Any] = {}
        for parents, _, step in self.walk():
            config.update(
                **prefix_keys(
                    step.config,
                    prefix=":".join([p.name for p in parents] + [step.name]) + ":",
                )
                if step.config is not None
                else {},
            )
        return config

    def configure(
        self,
        config: Config,
        *,
        rename: bool | str = False,
        prefixed_name: bool = False,
        transform_context: Any | None = None,
        params: Mapping[str, Any] | None = None,
        clear_space: bool | Literal["auto"] = "auto",
    ) -> Pipeline:
        """Configure the pipeline with the given configuration.

        This takes a pipeline with spaces and choices and trims it down based on the
        configuration. For example, choosing selected steps and setting the `config`
        of steps with those present in the `config` object given to this function.

        If there are any modules attached to this pipeline,
        these will also be configured for you.

        Args:
            config: The configuration to use
            rename: Whether to rename the pipeline. Defaults to `False`.
                * If `True`, the pipeline will be renamed using a random uuid
                * If a name is provided, the pipeline will be renamed to that name
                * If `False`, the pipeline will not be renamed
            prefixed_name: Whether the configuration is prefixed with the name of the
                pipeline. Defaults to `False`.
            transform_context: Any context to give to `config_transform=` of individual
                steps.
            params: The params to match any requests when configuring this step.
                These will match against any ParamRequests in the config and will
                be used to fill in any missing values.
            clear_space: Whether to clear the search space after configuring.
                If `"auto"` (default), then the search space will be cleared of any
                keys that are in the config, if the search space is a `dict`. Otherwise,
                `True` indicates that it will be removed in the returned step and
                `False` indicates that it will remain as is.

        Returns:
            A new pipeline with the configuration applied
        """
        this_config: Config
        if prefixed_name:
            this_config = mapping_select(config, f"{self.name}:")
        else:
            this_config = config

        config = dict(config)

        new_head = self.head.configure(
            this_config,
            transform_context=transform_context,
            params=params,
            prefixed_name=True,
            clear_space=clear_space,
        )

        new_modules = [
            module.configure(
                this_config,
                transform_context=transform_context,
                params=params,
                prefixed_name=True,
                clear_space=clear_space,
            )
            for module in self.modules.values()
        ]

        if rename is True:
            _rename = None
        elif rename is False:
            _rename = self.name
        else:
            _rename = rename

        return Pipeline.create(new_head, modules=new_modules, name=_rename)

    @overload
    def build(self, builder: None = None, **builder_kwargs: Any) -> Any:
        ...

    @overload
    def build(self, builder: Callable[[Pipeline], B], **builder_kwargs: Any) -> B:
        ...

    def build(
        self,
        builder: Callable[[Pipeline], B] | None = None,
        **builder_kwargs: Any,
    ) -> B | Any:
        """Build the pipeline.

        Args:
            builder: The builder to use. Default is `None`.
                * If `None` is provided, the assembler will attempt to automatically
                    figure out build the pipeline as it can.
                * If `builder` is a callable, we will attempt to use that.
            **builder_kwargs: Any additional keyword arguments to pass to the builder.

        Returns:
            The built pipeline
        """
        from amltk.building import build  # Prevent circular imports

        return build(self, builder=builder, **builder_kwargs)

    def copy(self, *, name: str | None = None) -> Pipeline:
        """Copy the pipeline.

        Returns:
            A copy of the pipeline
        """
        return self.create(self, name=self.name if name is None else name)

    @classmethod
    def create(
        cls,
        *steps: Step | Pipeline | Iterable[Step],
        modules: Pipeline | Step | Iterable[Pipeline | Step] | None = None,
        name: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Pipeline:
        """Create a pipeline from a sequence of steps.

        ??? warning "Using another pipeline `create()`"

            When using another pipeline as part of a substep, we handle
            the parameters of subpiplines in the following ways:

            * `modules`: Any modules attached to a subpipeline will be copied
                and attached to the new pipeline. If there is a naming conflict,
                an error will be raised.

            * `meta`: Any metadata associated with subpiplines will be erased.
                Please retrieve them an handle accordingly.

        Args:
            *steps: The steps to create the pipeline from
            name: The name of the pipeline. Defaults to a uuid
            modules: The modules to use for the pipeline
            meta: The meta information to attach to the pipeline

        Returns:
            Pipeline
        """
        # Expand out any pipelines in the init
        expanded = [s.steps if isinstance(s, Pipeline) else s for s in steps]
        step_sequence = list(Step.chain(*expanded))

        if name is None:
            name = str(uuid4())

        if isinstance(modules, (Pipeline, Step)):
            modules = [modules]

        # Collect all the modules, turning them into pipelines
        # as required by the internal api
        final_modules: dict[str, Pipeline | Step] = {}
        if modules is not None:
            final_modules = {module.name: module.copy() for module in modules}

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

        return cls(
            name=name,
            steps=step_sequence,
            modules=final_modules,
            meta=meta,
        )

    def _rich_iter(
        self,
        connect: TextType | None = None,  # noqa: ARG002
    ) -> Iterator[RenderableType]:
        """Used to make it more inline with steps."""
        yield self.__rich__()

    @override
    def __rich__(self) -> RenderableType:
        """Get the rich renderable for the pipeline."""
        from rich.console import Group as RichGroup
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.rule import Rule
        from rich.table import Table
        from rich.text import Text

        def _contents() -> Iterator[RenderableType]:
            # Things for this pipeline
            if self.meta is not None:
                table = Table.grid(padding=(0, 1), expand=False)
                table.add_row("meta", Pretty(self.meta))
                table.add_section()
                yield table

            connecter = Text("↓", style="bold", justify="center")
            # The main pipeline
            yield from self.head._rich_iter(connect=connecter)

            if any(self.modules):
                yield Rule(title="Modules", style=self.RICH_PANEL_BORDER_COLOR)

                # Any modules attached to this pipeline
                for module in self.modules.values():
                    yield from module._rich_iter(connect=connecter)

        clr = self.RICH_PANEL_BORDER_COLOR
        title = Text.assemble(
            (classname(self), f"{clr} bold"),
            "(",
            (self.name, f"{clr} italic"),
            ")",
            style="default",
            end="",
        )
        return Panel(
            RichGroup(*_contents()),
            title=title,
            title_align="left",
            expand=False,
            border_style=clr,
        )
