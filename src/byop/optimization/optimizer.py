"""Protocols for the optimization module."""
from __future__ import annotations

from typing import Protocol

from byop.optimization.trial import Info, Trial


class Optimizer(Protocol[Info]):
    """An optimizer protocol.

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """

    def tell(self, report: Trial.Report[Info]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report for a trial
        """
        ...

    def ask(self) -> Trial[Info]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
