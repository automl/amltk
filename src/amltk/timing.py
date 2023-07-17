"""Module for timing things."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any, Iterator, Literal, Mapping, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class Timer:
    """A timer for measuring the time between two events.

    Attributes:
        start_time: The time at which the timer was started.
        kind: The method of timing.
    """

    start_time: float
    kind: Timer.Kind

    class Kind(Enum):
        """An enum for the type of timer."""

        WALL = auto()
        CPU = auto()
        PROCESS = auto()
        NOTSET = auto()

        def __str__(self) -> str:
            return self.name.lower()

        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, key: Any) -> Timer.Kind:
            """Get the enum value from a string.

            Args:
                key: The string to convert.

            Returns:
                The enum value.
            """
            if isinstance(key, str):
                try:
                    return Timer.Kind[key.upper()]
                except KeyError:
                    return Timer.Kind.NOTSET

            return Timer.Kind.NOTSET

    class Unit(Enum):
        """An enum for the units of time."""

        SECONDS = auto()
        MILLISECONDS = auto()
        MICROSECONDS = auto()
        NANOSECONDS = auto()
        NOTSET = auto()

        def __str__(self) -> str:
            return self.name.lower()

        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, key: Any) -> Timer.Unit:
            """Get the enum value from a string.

            Args:
                key: The string to convert.

            Returns:
                The enum value.
            """
            if isinstance(key, str):
                try:
                    return Timer.Unit[key.upper()]
                except KeyError:
                    return Timer.Unit.NOTSET

            return Timer.Unit.NOTSET

    @dataclass
    class Interval:
        """A time interval.

        Attributes:
            start: The start time.
            end: The end time.
            kind: The type of timer used.
            unit: The unit of time.
        """

        start: float
        end: float
        kind: Timer.Kind
        unit: Timer.Unit

        @property
        def duration(self) -> float:
            """The duration of the time interval."""
            return self.end - self.start

        def to_dict(
            self,
            *,
            prefix: str = "",
            ensure_str: bool = True,
        ) -> dict[str, Any]:
            """Convert the time interval to a dictionary."""
            return {
                **{
                    f"{prefix}{k}": (str(v) if ensure_str else v)
                    for k, v in asdict(self).items()
                },
                f"{prefix}duration": self.duration,
            }

        @classmethod
        def from_dict(cls, d: Mapping[str, Any]) -> Timer.Interval:
            """Create a time interval from a dictionary."""
            return cls(
                start=d["start"],
                end=d["end"],
                kind=Timer.Kind.from_str(d["kind"]),
                unit=Timer.Unit.from_str(d["unit"]),
            )

        @classmethod
        def na(cls) -> Timer.Interval:
            """Create a time interval with all values set to `None`."""
            return cls(
                start=np.nan,
                end=np.nan,
                kind=Timer.Kind.NOTSET,
                unit=Timer.Unit.NOTSET,
            )

    @classmethod
    @contextmanager
    def time(
        cls,
        kind: Timer.Kind | Literal["cpu", "wall", "process"] = "wall",
    ) -> Iterator[Interval]:
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
        kind: Timer.Kind | Literal["cpu", "wall", "process"] = "wall",
    ) -> Timer:
        """Start a timer.

        Args:
            kind: The type of timer to use.

        Returns:
            The timer.
        """
        if kind in (Timer.Kind.WALL, "wall"):
            return Timer(time.time(), Timer.Kind.WALL)

        if kind in (Timer.Kind.CPU, "cpu"):
            return Timer(time.perf_counter(), Timer.Kind.CPU)

        if kind in (Timer.Kind.PROCESS, "process"):
            return Timer(time.process_time(), Timer.Kind.PROCESS)

        raise ValueError(f"Unknown timer type: {kind}")

    def stop(self) -> Interval:
        """Stop the timer.

        Returns:
            A tuple of the start time, end time, and duration.
        """
        if self.kind == Timer.Kind.WALL:
            end = time.time()
            return Timer.Interval(
                self.start_time,
                end,
                Timer.Kind.WALL,
                Timer.Unit.SECONDS,
            )

        if self.kind == Timer.Kind.CPU:
            end = time.perf_counter()
            return Timer.Interval(
                self.start_time,
                end,
                Timer.Kind.CPU,
                Timer.Unit.SECONDS,
            )

        if self.kind == Timer.Kind.PROCESS:
            end = time.process_time()
            return Timer.Interval(
                self.start_time,
                end,
                Timer.Kind.PROCESS,
                Timer.Unit.SECONDS,
            )

        raise ValueError(f"Unknown timer type: {self.kind}")
