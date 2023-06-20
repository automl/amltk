"""Module for pipeline building.

The `Builder` is responsble for taking a configured pipeline and
assembling it into a runnable pipeline. By default, this will try
some rough heuristics to determine what to build from your
configured [`Pipeline`][amltk.pipeline.Pipeline].
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

from more_itertools import first_true, seekable

from amltk.building._sklearn_builder import sklearn_builder
from amltk.exceptions import safe_map

if TYPE_CHECKING:
    from amltk.pipeline.pipeline import Pipeline

    B = TypeVar("B")

DEFAULT_BUILDERS: list[Callable[[Pipeline], Any]] = [sklearn_builder]


class BuildError(Exception):
    """Error when a pipeline could not be built."""


@overload
def build(pipeline: Pipeline, builder: None = None) -> Any:
    ...


@overload
def build(
    pipeline: Pipeline,
    builder: Callable[[Pipeline], B],
) -> B:
    ...


def build(
    pipeline: Pipeline,
    builder: Callable[[Pipeline], B] | None = None,
) -> B | Any:
    """Build a pipeline into a usable object.

    Args:
        pipeline: The pipeline to build
        builder: The builder to use. Defaults to `None` which will
            try to determine the best builder to use.

    Returns:
        The built pipeline
    """
    builders: list[Any]

    if builder is None:
        builders = DEFAULT_BUILDERS

    elif callable(builder):
        builders = [builder]

    else:
        raise NotImplementedError(f"Builder {builder} is not supported")

    def _build(_builder: Callable[[Pipeline], B]) -> B:
        return _builder(pipeline)

    results = seekable(safe_map(_build, builders))

    is_result = lambda r: not (isinstance(r, tuple) and isinstance(r[0], Exception))

    selected_built_pipeline = first_true(results, default=None, pred=is_result)

    # If we didn't manage to build a pipeline, iterate through
    # the errors and raise a ValueError
    if selected_built_pipeline is None:
        results.seek(0)  # Reset to start of the iterator
        msgs = "\n".join(
            f"{builder}: {err} \n" for builder, err in zip(builders, results)
        )
        raise BuildError(f"Could not build pipeline with any of the builders:\n{msgs}")

    assert not isinstance(selected_built_pipeline, Exception)
    return selected_built_pipeline
