"""Module for timing things."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Literal
from typing_extensions import assert_never


class TimeUnit(Enum):
    """An enum for the units of time."""

    SECONDS = auto()
    MILLISECONDS = auto()
    MICROSECONDS = auto()
    NANOSECONDS = auto()


class TimeKind(Enum):
    """An enum for the type of timer."""

    WALL = auto()
    CPU = auto()
    PROCESS = auto()


@dataclass
class Timer:
    """A timer for measuring the time between two events.

    Attributes:
        start_time: The time at which the timer was started.
        kind: The method of timing.
    """

    WALL: ClassVar[Literal[TimeKind.WALL]] = TimeKind.WALL
    CPU: ClassVar[Literal[TimeKind.CPU]] = TimeKind.CPU
    PROCESS: ClassVar[Literal[TimeKind.PROCESS]] = TimeKind.PROCESS

    units: ClassVar[type[TimeUnit]] = TimeUnit
    kinds: ClassVar[type[TimeKind]] = TimeKind

    start_time: float
    kind: TimeKind

    @classmethod
    def start(
        cls,
        kind: TimeKind | Literal["cpu", "wall", "process"] = TimeKind.CPU,
    ) -> Timer:
        """Start a timer.

        Args:
            kind: The type of timer to use.

        Returns:
            The timer.
        """
        if kind in (TimeKind.WALL, "wall"):
            return Timer(time.time(), TimeKind.WALL)

        if kind in (TimeKind.CPU, "cpu"):
            return Timer(time.perf_counter(), TimeKind.CPU)

        if kind in (TimeKind.PROCESS, "process"):
            return Timer(time.process_time(), TimeKind.PROCESS)

        raise ValueError(f"Unknown timer type: {kind}")

    def stop(self) -> TimeInterval:
        """Stop the timer.

        Returns:
            A tuple of the start time, end time, and duration.
        """
        if self.kind == TimeKind.WALL:
            end = time.time()
            return TimeInterval(self.start_time, end, TimeKind.WALL, TimeUnit.SECONDS)

        if self.kind == TimeKind.CPU:
            end = time.perf_counter()
            return TimeInterval(self.start_time, end, TimeKind.CPU, TimeUnit.SECONDS)

        if self.kind == TimeKind.PROCESS:
            end = time.process_time()
            return TimeInterval(
                self.start_time,
                end,
                TimeKind.PROCESS,
                TimeUnit.SECONDS,
            )

        # NOTE: this only seems to work with `match` statements from python 3.10
        assert_never(self.kind)  # type: ignore


@dataclass
class TimeInterval:
    """A time interval.

    Attributes:
        start: The start time.
        end: The end time.
        kind: The type of timer used.
        unit: The unit of time.
    """

    start: float
    end: float
    kind: TimeKind
    unit: TimeUnit

    @property
    def duration(self) -> float:
        """The duration of the time interval."""
        return self.end - self.start
