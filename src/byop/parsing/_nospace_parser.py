"""A parser for a pipeline which contains no hyperparameters."""
from __future__ import annotations

from typing import Any

from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split, Step
from byop.types import Seed

EMPTY_SPACE_INDICATORS: tuple[Any, ...] = (None, {}, [], ())


def space_required(step: Step | Component | Choice | Split) -> bool:
    """Whether a step requires a space."""
    # Choices always require a hyperparameter in a space of a pipeline as we
    # need to know which component to choose.
    if isinstance(step, Choice):
        return True

    return (
        isinstance(step, (Component, Split))
        and step.space not in EMPTY_SPACE_INDICATORS
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
    if any(space_required(step) for step in pipeline.traverse()):
        msg = (
            "Pipeline contains a step(s) which has either:"
            "\n * A space which is not a `None`."
            "\n * A `Choice` which requires a space to be formed."
            "\nThese are not supported by the NoSpaceParser."
        )
        raise ValueError(msg)
