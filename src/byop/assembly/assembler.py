"""Module for pipeline assembly.

The `Assembler` is the main workhorse for understanding your pipeline and
getting something useful out of it's abstract representation. By default,
the Assembler will try to `"auto"` what space and even how to put your
pipeline together, but these are just best efforts and there is no real
way to automatically know data should be passed from step to step, i.e.
does you step use sklearn style `fit` and `predict` or Pytorch `forward`?

# DOC: Should document how this process actually works
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, overload

from attrs import frozen
from more_itertools import first, partition
from result import as_result

from byop.pipeline import Pipeline
from byop.spaces.parsers import DEFAULT_PARSERS, ParserFunction
from byop.typing import Key, Name, Seed, Space

# NOTE: See in `NOTES.md` about the `@overload` spam and `.pyi` files
# NOTE: It's important to keep this in the `TYPE_CHECKING` block because
# we can't assume it's installed.
if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


@frozen(kw_only=True)
class Assembler(Generic[Key, Name, Space]):
    """Wraps a pipeline to be able to assemble the different parts of it.

    Namely:
    * space: Extracting the configuration space from the pipeline
    * assembly: How to build the pipeline from a configuration
    """

    pipeline: Pipeline[Key, Name]
    space: Space

    @classmethod
    @overload
    def create(
        cls,
        pipeline: Pipeline[Key, Name],
        space: Callable[[Pipeline[Key, Name], Seed | None], Space],
    ) -> Assembler[Key, Name, Space]:
        ...

    @classmethod
    @overload
    def create(
        cls, pipeline: Pipeline[Key, Name], space: Literal["auto"]
    ) -> Assembler[Key, Name, Any]:
        ...

    @classmethod
    @overload
    def create(
        cls, pipeline: Pipeline[Key, Name], space: type[ConfigurationSpace]
    ) -> Assembler[Key, Name, ConfigurationSpace]:
        ...

    @classmethod
    def create(
        cls,
        pipeline: Pipeline[Key, Name],
        space: (
            Literal["auto"]
            | type[ConfigurationSpace]
            | Callable[[Pipeline[Key, Name], Seed | None], Space]
        ) = "auto",
        seed: Seed | None = None,
    ) -> Assembler[Key, Name, Any | Space]:
        """Create an assembler for a pipeline.

        Args:
            pipeline: The pipeline to assemble
            space: The space to use for the assembler. Default is `"auto"`

                * If `"auto"` is provided, the assembler will attempt to
                automatically figure out the kind of Space to extract from the pipeline.

                If a `space` is a type, we will match to this any parser that i
                capable of parsing this type of space

                If a `space` is a callable, we will use this callable to parse the
                pipeline.
                # DOC: Shuold be extended
            seed: The seed to seed the space with if applicable. Defaults to `None`

        Returns:
            Assembler
        """
        # Order is relevant here
        parsers: list[ParserFunction]
        if space == "auto":
            parsers = [p.parse for p in DEFAULT_PARSERS]
        elif isinstance(space, type):
            parsers = [p.parse for p in DEFAULT_PARSERS if p.supports(space)]
        elif callable(space):
            parsers = [as_result(Exception)(space)]
        else:
            raise NotImplementedError(f"Unknown what to do with {space=}")

        parse_attmptes = (parse(pipeline, seed) for parse in parsers)
        valid_parses, errs = partition(lambda r: r.is_ok(), parse_attmptes)
        selected_space = first(valid_parses, default=None)

        if selected_space is None:
            raise ValueError(
                "Could not create a space from your pipeline with the parsers",
                f" {parsers=}\nParser errors\n{errs=}",
            )
        return Assembler(pipeline=pipeline, space=selected_space)
