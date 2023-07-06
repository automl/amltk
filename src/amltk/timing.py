"""Module for timing things."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Iterator, Literal, Mapping, TypeVar
from typing_extensions import assert_never

import numpy as np

T = TypeVar("T")


class TimeUnit(Enum):
    """An enum for the units of time."""

    SECONDS = auto()
    MILLISECONDS = auto()
    MICROSECONDS = auto()
    NANOSECONDS = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.name.lower()

    @classmethod
    def get(cls, key: Any) -> TimeUnit:
        """Get the enum value from a string.

        Args:
            key: The string to convert.

        Returns:
            The enum value.
        """
        if isinstance(key, str):
            try:
                return TimeUnit[key.upper()]
            except KeyError:
                return TimeUnit.UNKNOWN

        return TimeUnit.UNKNOWN


class TimeKind(Enum):
    """An enum for the type of timer."""

    WALL = auto()
    CPU = auto()
    PROCESS = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.name.lower()

    @classmethod
    def get(cls, key: Any) -> TimeKind:
        """Get the enum value from a string.

        Args:
            key: The string to convert.

        Returns:
            The enum value.
        """
        if isinstance(key, str):
            try:
                return TimeKind[key.upper()]
            except KeyError:
                return TimeKind.UNKNOWN

        return TimeKind.UNKNOWN


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

    @contextmanager
    @classmethod
    def time(
        cls,
        kind: TimeKind | Literal["cpu", "wall", "process"] = TimeKind.WALL,
    ) -> Iterator[TimeInterval]:
        """Time a block of code.

        Args:
            kind: The type of timer to use.

        Yields:
            The timer.
        """
        timer = cls.start(kind=kind)
        interval = timer.stop()
        yield interval
        _interval = timer.stop()
        interval.end = _interval.end

    @classmethod
    def start(
        cls,
        kind: TimeKind | Literal["cpu", "wall", "process"] = TimeKind.WALL,
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

    def to_dict(self, *, prefix: str = "") -> dict[str, Any]:
        """Convert the time interval to a dictionary."""
        return {
            **{f"{prefix}{k}": v for k, v in asdict(self).items()},
            f"{prefix}duration": self.duration,
        }

    def dict_for_dataframe(self) -> dict[str, Any]:
        """Convert the time interval to a dictionary for a dataframe."""
        return {
            "start": self.start,
            "end": self.end,
            "kind": str(self.kind),
            "unit": str(self.unit),
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TimeInterval:
        """Create a time interval from a dictionary."""
        return cls(
            start=d["start"],
            end=d["end"],
            kind=TimeKind.get(d["kind"]),
            unit=TimeUnit.get(d["unit"]),
        )

    @classmethod
    def na_time_interval(cls) -> TimeInterval:
        """Create a time interval with all values set to `None`."""
        return cls(
            start=np.nan,
            end=np.nan,
            kind=TimeKind.UNKNOWN,
            unit=TimeUnit.UNKNOWN,
        )
