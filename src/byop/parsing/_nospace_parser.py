"""A parser for a pipeline which contains no hyperparameters."""
from __future__ import annotations

from typing import Any

from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split, Step
from byop.types import Seed

EMPTY_SPACE_INDICATORS: tuple[Any, ...] = (None, {}, [], ())


def space_required(step: Pipeline | Step | Component | Choice | Split) -> bool:
    """Whether a step requires a space."""
    # Choices always require a hyperparameter in a space of a pipeline as we
    # need to know which component to choose.
    if isinstance(step, Pipeline):
        return any(space_required(s) for s in step.traverse())

    if isinstance(step, Choice):
        return True

    return (
        isinstance(step, (Component, Split))
        and step.search_space not in EMPTY_SPACE_INDICATORS
    )


def nospace_parser(
    pipeline: Pipeline,
    seed: Seed | None = None,  # noqa: ARG001
) -> None:
    """Parse a pipeline which contains no hyperparameters.

    Args:
        pipeline: The pipeline to parse
        seed: The seed to use for the parsing

    Returns:
        Nothing if successful
    """
    if (
        any(space_required(step) for step in pipeline.traverse())
        or any(space_required(module) for module in pipeline.modules.values())
        or any(space_required(s) for s in pipeline.searchables.values())
    ):
        msg = (
            "Pipeline contains a step(s)/module(s)/searchable(s) which has either:"
            "\n * A space which is not a `None`."
            "\n * A `Choice` which requires a space to be formed."
            "\nThese are not supported by the NoSpaceParser."
        )
        raise ValueError(msg)
