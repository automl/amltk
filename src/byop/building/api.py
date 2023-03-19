"""Module for pipeline building.

The `Builder` is responsble for taking a configured pipeline and
assembling it into a runnable pipeline. By default, this will try
"auto" which uses some rough heuristics to determine what to build
from your configured [`Pipeline`][byop.pipeline.Pipeline].
"""
from __future__ import annotations

from typing import Any, Callable, Literal, overload

from more_itertools import first_true, seekable
from result import Result, as_result

from byop.building.builders import DEFAULT_BUILDERS, Builder
from byop.pipeline.pipeline import Pipeline
from byop.types import BuiltPipeline


@overload
def build(pipeline: Pipeline) -> Any:
    ...


@overload
def build(pipeline: Pipeline, builder: Literal["auto"]) -> Any:
    ...


@overload
def build(pipeline: Pipeline, builder: Builder[BuiltPipeline]) -> BuiltPipeline:
    ...


@overload
def build(
    pipeline: Pipeline,
    builder: Callable[[Pipeline], BuiltPipeline],
) -> BuiltPipeline:
    ...


def build(
    pipeline: Pipeline,
    builder: (
        Literal["auto"] | Builder[BuiltPipeline] | Callable[[Pipeline], BuiltPipeline]
    ) = "auto",
) -> BuiltPipeline | Any:
    """Build a pipeline into a usable object.

    Args:
        pipeline: The pipeline to build
        builder: The builder to use. Defaults to "auto" which will
            try to determine the best builder to use.

    Returns:
        The built pipeline
    """
    results: seekable[Result[BuiltPipeline, Exception]]
    builders: list[Any]

    if builder == "auto":
        builders = DEFAULT_BUILDERS
        results = seekable(builder._build(pipeline) for builder in builders)

    elif isinstance(builder, Builder):
        builders = [builder]
        results = seekable(builder._build(pipeline) for builder in builders)

    elif callable(builder):
        builders = [builder]
        safe_builder = as_result(Exception)(builder)
        results = seekable([safe_builder(pipeline)])

    else:
        raise NotImplementedError(f"Builder {builder} is not supported")

    selected_built_pipeline = first_true(
        results,
        default=None,
        pred=lambda r: r.is_ok(),
    )

    if selected_built_pipeline is None:
        results.seek(0)  # Reset to start of the iterator
        errs = [r.unwrap_err() for r in results]
        raise ValueError(
            "Could not build a pipeline with any of the builders:"
            f" {builders=}\nBuilder errors\n{errs=}",
        )

    assert selected_built_pipeline.is_ok()
    return selected_built_pipeline.unwrap()
