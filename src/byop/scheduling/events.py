"""A module containing events for a scheduler."""
from __future__ import annotations

from enum import Enum, auto


class TaskStatus(Enum):
    """The status of a task."""

    SUBMITTED = 0  # Submitted
    PENDING = auto()  # Pending execution
    RUNNING = auto()  # Running
    ERROR = auto()  # Failure
    UPDATE = auto()  # Updated
    FINISHED = auto()  # Has ended
    WAITING = auto()  # Waiting a response in comms
    SUCCESS = auto()  # Success
    CANCELLED = auto()  # Cancelled


class SchedulerStatus(Enum):
    """The state of a controller."""

    STARTED = 100
    RUNNING = auto()
    STOPPING = auto()
    FINISHING = auto()
    FINISHED = auto()


class ExitCode(Enum):
    """The reason the scheduler ended."""

    STOPPED = 10_000  # `stop` was called
    TIMEOUT = auto()  # Timeout was hit
    EMPTY = auto()  # Queue and results were empty
    UNKNOWN = auto()  # Unknown reason for stopping
