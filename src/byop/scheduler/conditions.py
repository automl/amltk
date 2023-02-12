"""A module for defining delayed conditions to interact with the scheduler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from byop.fluid import DelayedOp
from byop.scheduler.events import SchedulerStatus, TaskStatus

if TYPE_CHECKING:
    from byop.scheduler.scheduler import Scheduler

Event = TypeVar("Event", TaskStatus, SchedulerStatus)
X = TypeVar("X")


@dataclass
class SchedulerCountCondition:
    """A class for defining easy conditionals to check on a scheduler."""

    scheduler: Scheduler

    def __call__(self, event: Event) -> DelayedOp[int, ...]:
        """Create a predicate for a count of a specific event."""

        def count(*args: Any, **kwargs: Any) -> int:  # noqa: ARG001
            return self.scheduler.events.count[event]

        return DelayedOp(count)
