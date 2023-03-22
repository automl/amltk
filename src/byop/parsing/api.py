"""Module for pipeline parsing.

The `Parser` is the main workhorse for understanding your pipeline and
getting something useful out of it's abstract representation. By default,
the `Parser` will try to `"auto"` what space to extract.

# DOC: Should document how this process actually works
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from more_itertools import first_true, seekable

from byop.exceptions import attach_traceback
from byop.parsing._configspace_parser import configspace_parser
from byop.parsing._nospace_parser import nospace_parser
from byop.parsing._optuna_parser import optuna_parser
from byop.types import Seed, Space, safe_issubclass

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from byop.pipeline import Pipeline  # Prevent recursive imports

DEFAULT_PARSERS: list[
    Callable[[Pipeline], Any] | Callable[[Pipeline, Seed | None], Any]
] = [
    nospace_parser,  # Retain this as the first one
    configspace_parser,
    optuna_parser,
]

SENTINAL = object()


class ParseError(Exception):
    """Error when a pipeline could not be parsed."""


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


# Call with a optuna literal
@overload
def parse(
    pipeline: Pipeline,
    parser: Literal["optuna"] = "optuna",
    *,
    seed: Seed | None = ...,
) -> Any:
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


def parse(
    pipeline: Pipeline,
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
) -> Space | ConfigurationSpace | Any:
    """Create a space from a pipeline.

    Args:
        pipeline: The pipeline to parse.
        parser: The parser to use for assembling the space. Default is `"auto"`.
            * If `"auto"` is provided, the assembler will attempt to
            automatically figure out the kind of Space to extract from the pipeline.
            * If `"configspace"` is provided, a ConfigurationSpace will be attempted
            to be extracted.
            * If `"optuna"` is provided, an Optuna space will be attempted
            to be extracted.
            * If a `type` is provided, it will attempt to infer which parser to use.
            * If `parser` is a parser type, we will attempt to use that.
            * If `parser` is a callable, we will attempt to use that.
            If there are other intuitive ways to indicate the type, please open
            an issue on GitHub and we will consider it!
        seed: The seed to seed the space with if applicable.

    Returns:
        The built space
    """
    parsers: list[Any]

    if parser == "auto":
        parsers = DEFAULT_PARSERS
    elif parser == "configspace" or (
        isinstance(parser, type) and safe_issubclass(parser, "ConfigurationSpace")
    ):
        parsers = [configspace_parser]
    elif parser == "optuna":
        parsers = [optuna_parser]
    elif callable(parser):
        parsers = [parser]
    else:
        raise NotImplementedError(f"Unknown what to do with {parser=}")

    def _parse(
        _parser: Callable[[Pipeline], Space] | Callable[[Pipeline, Seed], Space],
        _pipeline: Pipeline,
        _seed: Seed | None = None,
    ) -> Space | Exception:
        """Attempt to parse a pipeline with a parser, catching any errors."""
        try:
            if _seed is not None:
                return _parser(_pipeline, _seed)  # type: ignore
            return _parser(_pipeline)  # type: ignore
        except Exception as e:  # noqa: BLE001
            return attach_traceback(e)

    itr = (_parse(_parser, pipeline, seed) for _parser in parsers)
    results = seekable(itr)
    selected_space = first_true(
        results,
        default=SENTINAL,
        pred=lambda r: not isinstance(r, Exception),
    )

    # If we didn't manage to parse a space,
    # iterate through the errors and raise a ValueError
    if selected_space is SENTINAL:
        results.seek(0)  # Reset to start of the iterator
        msgs = "\n".join(f"{parser}: {err}" for parser, err in zip(parsers, results))
        raise ParseError(f"Could not parse pipeline with any of the parser:\n{msgs}")

    assert not isinstance(selected_space, Exception)
    return selected_space
