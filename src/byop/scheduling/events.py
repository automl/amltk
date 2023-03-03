"""A module containing events and exit codes for a scheduler and tasks."""
from __future__ import annotations

from enum import Enum, auto
from typing import Union

from typing_extensions import TypeAlias

from byop.types import TaskName


class TaskEvent(Enum):
    """The status of a task."""

    SUBMITTED = auto()
    """A Task has been submitted to the scheduler."""

    DONE = auto()
    """A Task has finished running."""

    CANCELLED = auto()
    """A Task has been cancelled."""

    RETURNED = auto()
    """A Task has successfully returned a value."""

    NO_RETURN = auto()
    """A Task failed to return anything."""

    UPDATE = auto()
    """A CommTask has sent an update with `send`."""

    WAITING = auto()
    """A CommTask is waiting for a response to `recv`."""


class SchedulerEvent(Enum):
    """The state of a controller."""

    STARTED = auto()
    """The scheduler has started.

    This means the scheduler has started up the executor and is ready to
    start deploying tasks to the executor.
    """

    FINISHING = auto()
    """The scheduler is finishing.

    This means the executor is still running but the stopping criterion
    for the scheduler are no longer monitored. If using `run(..., wait=True)`
    which is the deafult, the scheduler will wait until the queue as been
    emptied before reaching STOPPED.
    """

    FINISHED = auto()
    """The scheduler has finished.

    This means the scheduler has stopped running the executor and
    has processed all futures and events. This is the last event
    that will be emitted from the scheduler before ceasing.
    """

    STOP = auto()
    """The scheduler has been stopped.

    This means the executor is no longer running so no further tasks can be
    dispatched. The scheduler is in a state where it will wait for the current
    queue to empty out (if `run(..., wait=True)`) and for any futures to be
    processed.
    """

    TIMEOUT = auto()
    """The scheduler has reached the timeout.

    This means the scheduler reached the timeout stopping criterion, which
    is only active when `run(..., timeout: float)` was used to start the
    scheduler.
    """

    EMPTY = auto()
    """The scheduler has an empty queue.

    This means the scheduler has no more running tasks in it's queue.
    This event will only trigger when `run(..., end_on_empty=False)`
    was used to start the scheduler.
    """


class ExitCode(Enum):
    """The reason the scheduler ended."""

    STOPPED = auto()
    """The scheduler was stopped forcefully with `Scheduler.stop`."""

    TIMEOUT = auto()
    """The scheduler finished because of a timeout."""

    EXHAUSTED = auto()
    """The scheduler finished because it exhausted its queue."""

    UNKNOWN = auto()
    """The scheduler finished for an unknown reason."""


EventTypes: TypeAlias = Union[TaskEvent, SchedulerEvent, tuple[TaskName, TaskEvent]]
"""Any possible event time emitted by the scheduler."""
