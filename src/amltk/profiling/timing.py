"""Module for timing things."""
from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd

from amltk._functional import dict_get_not_none

if TYPE_CHECKING:
    from pandas._libs.missing import NAType


@dataclass
class Timer:
    """A timer for measuring the time between two events.

    Attributes:
        start_time: The time at which the timer was started.
        kind: The method of timing.
    """

    start_time: float
    kind: Timer.Kind | NAType

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Timer.Interval:
        """Create a time interval from a dictionary."""
        return Timer.Interval(
            start=dict_get_not_none(d, "start", np.nan),
            end=dict_get_not_none(d, "end", np.nan),
            kind=Timer.Kind.from_str(dict_get_not_none(d, "kind", pd.NA)),
            unit=Timer.Unit.from_str(dict_get_not_none(d, "unit", pd.NA)),
        )

    @classmethod
    def na(cls) -> Timer.Interval:
        """Create a time interval with all values set to `None`."""
        return Timer.Interval(
            start=np.nan,
            end=np.nan,
            kind=pd.NA,
            unit=pd.NA,
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
            The Time Interval. The start and end times will not be
            valid until the context manager is exited.
        """
        timer = cls.start(kind=kind)

        interval = Timer.na()
        interval.kind = timer.kind
        interval.unit = Timer.Unit.SECONDS

        try:
            yield interval
        finally:
            _interval = timer.stop()
            interval.start = _interval.start
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
        kind: Timer.Kind | NAType
        unit: Timer.Unit | NAType

        @property
        def duration(self) -> float:
            """The duration of the time interval."""
            return self.end - self.start

        def to_dict(self, *, prefix: str = "") -> dict[str, Any]:
            """Convert the time interval to a dictionary."""
            return {
                f"{prefix}start": self.start,
                f"{prefix}end": self.end,
                f"{prefix}duration": self.duration,
                f"{prefix}kind": self.kind,
                f"{prefix}unit": self.unit,
            }

    class Kind(str, Enum):
        """An enum for the type of timer."""

        WALL = "wall"
        CPU = "cpu"
        PROCESS = "process"

        @override
        def __str__(self) -> str:
            return self.name.lower()

        @override
        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, key: Any) -> Timer.Kind | NAType:
            """Get the enum value from a string.

            Args:
                key: The string to convert.

            Returns:
                The enum value.
            """
            if isinstance(key, Timer.Kind):
                return key

            if isinstance(key, str):
                try:
                    return Timer.Kind[key.upper()]
                except KeyError:
                    return pd.NA

            return pd.NA

    class Unit(str, Enum):
        """An enum for the units of time."""

        SECONDS = "seconds"
        MILLISECONDS = "milliseconds"
        MICROSECONDS = "microseconds"
        NANOSECONDS = "nanoseconds"

        @override
        def __str__(self) -> str:
            return self.name.lower()

        @override
        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, key: Any) -> Timer.Unit | NAType:
            """Get the enum value from a string.

            Args:
                key: The string to convert.

            Returns:
                The enum value.
            """
            if isinstance(key, Timer.Unit):
                return key

            if isinstance(key, str):
                try:
                    return Timer.Unit[key.upper()]
                except KeyError:
                    return pd.NA

            return pd.NA
