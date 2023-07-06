"""Protocols for the optimization module."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from amltk.optimization.trial import Trial

I = TypeVar("I")  # noqa: E741


class Optimizer(Generic[I]):
    """An optimizer protocol.

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """

    @abstractmethod
    def tell(self, report: Trial.Report[I]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report for a trial
        """

    @abstractmethod
    def ask(self) -> Trial[I]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
