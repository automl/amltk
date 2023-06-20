"""Protocols for the optimization module."""
from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from amltk.optimization.trial import Info, Trial


class Optimizer(Generic[Info]):
    """An optimizer protocol.

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """

    @abstractmethod
    def tell(self, report: Trial.Report[Info]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report for a trial
        """

    @abstractmethod
    def ask(self) -> Trial[Info]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
