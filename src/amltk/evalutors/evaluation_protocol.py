"""Evaluation protocols for how a trial and a pipeline should be evaluated.

TODO: Sorry
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from amltk.scheduling import Plugin

if TYPE_CHECKING:
    from amltk.optimization import Trial
    from amltk.pipeline import Node
    from amltk.scheduling import Scheduler, Task


class EvaluationProtocol:
    """A protocol for how a trial should be evaluated on a pipeline."""

    fn: Callable[[Trial, Node], Trial.Report]

    def task(
        self,
        scheduler: Scheduler,
        plugins: Plugin | Iterable[Plugin] | None = None,
    ) -> Task[[Trial, Node], Trial.Report]:
        """Create a task for this protocol.

        Args:
            scheduler: The scheduler to use for the task.
            plugins: The plugins to use for the task.

        Returns:
            The created task.
        """
        _plugins: tuple[Plugin, ...]
        match plugins:
            case None:
                _plugins = ()
            case Plugin():
                _plugins = (plugins,)
            case Iterable():
                _plugins = tuple(plugins)

        return scheduler.task(self.fn, plugins=_plugins)


class CustomProtocol(EvaluationProtocol):
    """A custom evaluation protocol based on a user function."""

    def __init__(self, fn: Callable[[Trial, Node], Trial.Report]) -> None:
        """Initialize the protocol.

        Args:
            fn: The function to use for the evaluation.
        """
        super().__init__()
        self.fn = fn
