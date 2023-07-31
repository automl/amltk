"""Module for pipeline building.

The `Builder` is responsble for taking a configured pipeline and
assembling it into a runnable pipeline. By default, this will try
some rough heuristics to determine what to build from your
configured [`Pipeline`][amltk.pipeline.Pipeline].
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from typing_extensions import override

from more_itertools import first_true, seekable

from amltk.building._sklearn_builder import sklearn_builder
from amltk.exceptions import safe_map
from amltk.functional import funcname

if TYPE_CHECKING:
    from amltk.pipeline.pipeline import Pipeline

    B = TypeVar("B")

DEFAULT_BUILDERS: list[Callable[[Pipeline], Any]] = [sklearn_builder]

logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Error when a pipeline could not be built."""

    def __init__(
        self,
        builders: list[str],
        err_tbs: list[tuple[Exception, str]],
    ) -> None:
        """Create a new BuildError.

        Args:
            builders: The builders that were tried
            err_tbs: The errors and tracebacks for each builder
        """
        self.builders = builders
        self.err_tbs = err_tbs
        super().__init__(builders, err_tbs)

    @override
    def __str__(self) -> str:
        return "\n".join(
            [
                "Could not build pipeline with any of the builders:",
                *[
                    f"  - {builder}: {err}\n{tb}"
                    for builder, (err, tb) in zip(self.builders, self.err_tbs)
                ],
            ],
        )


@overload
def build(pipeline: Pipeline, builder: None = None, **builder_kwargs: Any) -> Any:
    ...


@overload
def build(
    pipeline: Pipeline,
    builder: Callable[[Pipeline], B],
    **builder_kwargs: Any,
) -> B:
    ...


def build(
    pipeline: Pipeline,
    builder: Callable[[Pipeline], B] | None = None,
    **builder_kwargs: Any,
) -> B | Any:
    """Build a pipeline into a usable object.

    Args:
        pipeline: The pipeline to build
        builder: The builder to use. Defaults to `None` which will
            try to determine the best builder to use.
        **builder_kwargs: Any keyword arguments to pass to the builder

    Returns:
        The built pipeline
    """
    builders: list[Any]

    if builder is None:
        builders = DEFAULT_BUILDERS
        if any(builder_kwargs):
            logger.warning(
                f"If using `{builder_kwargs=}`, you most likely want to"
                " pass an explicit `builder` argument",
            )

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
        builders = [funcname(builder) for builder in builders]
        errors = [(err, tb) for err, tb in results]  # type: ignore
        raise BuildError(builders=builders, err_tbs=errors)

    assert not isinstance(selected_built_pipeline, Exception)
    return selected_built_pipeline
