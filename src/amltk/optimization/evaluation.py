"""Evaluation protocols for how a trial and a pipeline should be evaluated.

TODO: Sorry
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from amltk.optimization import Trial
    from amltk.pipeline import Node


@runtime_checkable
class EvaluationProtocol(Protocol):
    """A protocol for how a trial should be evaluated on a pipeline."""

    fn: Callable[[Trial, Node], Trial.Report]


class CustomEvaluationProtocol(EvaluationProtocol):
    """A custom evaluation protocol based on a user function."""

    def __init__(self, fn: Callable[[Trial, Node], Trial.Report]) -> None:
        """Initialize the protocol.

        Args:
            fn: The function to use for the evaluation.
        """
        super().__init__()
        self.fn = fn
