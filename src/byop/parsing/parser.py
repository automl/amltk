"""Module for pipeline parsing.

The `Parser` is the main workhorse for understanding your pipeline and
getting something useful out of it's abstract representation. By default,
the `Parser` will try to `"auto"` what space to extract.

# DOC: Should document how this process actually works
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, overload

from more_itertools import first_true, seekable
from result import Result, as_result
from typing_extensions import TypeAlias

from byop.parsing.space_parsers import DEFAULT_PARSERS
from byop.pipeline import Pipeline
from byop.typing import Seed, Space

# NOTE: It's important to keep this in the `TYPE_CHECKING` block because
# we can't assume it's installed.
if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

ParserFunctionNoSeed: TypeAlias = Callable[[Pipeline], Result[Space, Exception]]
ParserFunctionSeed: TypeAlias = Callable[
    [Pipeline, Optional[Seed]],
    Result[Space, Exception],
]
ParserFunction = Union[ParserFunctionNoSeed, ParserFunctionSeed]

# Simple call
@overload
def parse(pipeline: Pipeline) -> Any:
    ...


# Auto parsing, with or without seed
@overload
def parse(
    pipeline: Pipeline,
    space: Literal["auto"] = "auto",
    *,
    seed: Seed | None = ...,
) -> Any:
    ...


# Call with callable to parse, accepting or not accepting a seed
@overload
def parse(
    pipeline: Pipeline,
    space: Callable[[Pipeline], Space] | Callable[[Pipeline, Seed | None], Space],
    *,
    seed: Seed | None = ...,
) -> Space:
    ...


# Call with ConfigurationSpace type as argument
@overload
def parse(
    pipeline: Pipeline,
    space: type[ConfigurationSpace],
    *,
    seed: Seed | None = ...,
) -> ConfigurationSpace:
    ...


def parse(
    pipeline: Pipeline,
    space: (
        Literal["auto"]
        | type[ConfigurationSpace]
        | Callable[[Pipeline], Space]
        | Callable[[Pipeline, Seed | None], Space]
    ) = "auto",
    *,
    seed: Seed | None = None,
) -> Space | ConfigurationSpace | Any:
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
        Space | ConfigurationSpace | Any
            The built space
    """
    # Order is relevant here
    results: seekable[Result[Space, Exception]]
    parsers: list[Any]

    if space == "auto":
        parsers = DEFAULT_PARSERS
        results = seekable(parser.parse(pipeline, seed) for parser in parsers)

    elif isinstance(space, type):
        parsers = [parser for parser in DEFAULT_PARSERS if parser.supports(space)]
        results = seekable(parser.parse(pipeline, seed) for parser in parsers)

    elif callable(space) and seed is not None:
        parsers = [space]
        safe_space = as_result(Exception)(space)  # type: ignore
        results = seekable([safe_space(pipeline, seed)])  # type: ignore

    elif callable(space) and seed is None:
        parsers = [space]
        safe_space = as_result(Exception)(space)  # type: ignore
        results = seekable([safe_space(pipeline)])  # type: ignore

    else:
        raise NotImplementedError(f"Unknown what to do with {space=}")

    selected_space = first_true(results, default=None, pred=lambda r: r.is_ok())

    if selected_space is None:
        results.seek(0)  # Reset to start of the iterator
        errs = [r.unwrap_err() for r in results]
        raise ValueError(
            "Could not create a space from your pipeline with the parsers",
            f" {parsers=}\nParser errors\n{errs=}",
        )

    assert selected_space.is_ok()
    return selected_space.unwrap()
