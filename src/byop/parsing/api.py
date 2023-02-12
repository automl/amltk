"""Module for pipeline parsing.

The `Parser` is the main workhorse for understanding your pipeline and
getting something useful out of it's abstract representation. By default,
the `Parser` will try to `"auto"` what space to extract.

# DOC: Should document how this process actually works
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from more_itertools import first_true, seekable
from result import Result, as_result

from byop.parsing.space_parsers import DEFAULT_PARSERS, ParseError, SpaceParser
from byop.parsing.space_parsers.configspace import ConfigSpaceParser
from byop.types import Seed, Space, safe_issubclass

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from byop.pipeline import Pipeline  # Prevent recursive imports


# Auto parsing, with or without seed
@overload
def parse(
    pipeline: Pipeline,
    parser: Literal["auto"] = "auto",
    *,
    seed: Seed | None = ...,
) -> Any:
    ...


# Call with a configspace literal
@overload
def parse(
    pipeline: Pipeline,
    parser: Literal["configspace"] | type[ConfigurationSpace],
    *,
    seed: Seed | None = ...,
) -> ConfigurationSpace:
    ...


# Call with callable to parse, accepting or not accepting a seed
@overload
def parse(
    pipeline: Pipeline,
    parser: Callable[[Pipeline], Space] | Callable[[Pipeline, Seed | None], Space],
    *,
    seed: Seed | None = ...,
) -> Space:
    ...


# Call with a parser
@overload
def parse(
    pipeline: Pipeline,
    parser: SpaceParser[Space] | type[SpaceParser[Space]],
    *,
    seed: Seed | None = ...,
) -> Space:
    ...


def parse(
    pipeline: Pipeline,
    parser: (
        Literal["auto"]
        | Literal["configspace"]
        | type[ConfigurationSpace]
        | Callable[[Pipeline], Space]
        | Callable[[Pipeline, Seed | None], Space]
        | SpaceParser[Space]
        | type[SpaceParser[Space]]
    ) = "auto",
    *,
    seed: Seed | None = None,
) -> Space | ConfigurationSpace | Any:
    """Create a space from a pipeline.

    Args:
        pipeline: The pipeline to parse.
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
        Space | ConfigurationSpace | Any
            The built space
    """
    results: seekable[Result[Space, ParseError | Exception]]
    parsers: list[Any]

    if parser == "auto":
        parsers = DEFAULT_PARSERS
        results = seekable(p._parse(pipeline, seed) for p in DEFAULT_PARSERS)

    elif parser == "configspace":
        parsers = [ConfigSpaceParser]
        results = seekable([ConfigSpaceParser._parse(pipeline, seed)])

    elif isinstance(parser, type) and safe_issubclass(parser, "ConfigurationSpace"):
        parsers = [ConfigSpaceParser]
        results = seekable([ConfigSpaceParser._parse(pipeline, seed)])

    elif isinstance(parser, SpaceParser):
        parsers = [parser]
        results = seekable([parser._parse(pipeline, seed)])

    elif callable(parser) and seed is not None:
        parsers = [parser]
        safe_space = as_result(Exception)(parser)  # type: ignore
        results = seekable([safe_space(pipeline, seed)])  # type: ignore

    elif callable(parser) and seed is None:
        parsers = [parser]
        safe_space = as_result(Exception)(parser)  # type: ignore
        results = seekable([safe_space(pipeline)])  # type: ignore

    else:
        raise NotImplementedError(f"Unknown what to do with {parser=}")

    selected_space = first_true(results, default=None, pred=lambda r: r.is_ok())

    if selected_space is None:
        results.seek(0)  # Reset to start of the iterator
        errs = [r.unwrap_err() for r in results]
        parser_errs = [e for e in errs if isinstance(e, ParseError)]
        others = [e for e in errs if not isinstance(e, ParseError)]
        msg = (
            "Could not create a space from your pipeline with the parsers"
            f"\n{parsers=}"
            "\n"
            f" Parser errors:\n{parser_errs=}"
            f" Other errors:\n{others=}"
        )
        raise ParseError(msg)

    assert selected_space.is_ok()
    return selected_space.unwrap()
