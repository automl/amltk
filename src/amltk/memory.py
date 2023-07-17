"""Module to measure memory."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any, Iterator, Literal, Mapping, TypeVar

import psutil

T = TypeVar("T")


@dataclass
class Memory:
    """A timer for measuring the time between two events.

    Attributes:
        start_time: The time at which the timer was started.
        kind: The method of timing.
    """

    start_memory: float
    kind: Memory.Kind
    unit: Memory.Unit

    @dataclass
    class Interval:
        """A class for representing a time interval.

        Attributes:
            before: The memory before the interval.
            after: The memory after the interval.
            kind: The kind of memory.
            unit: The unit of memory.
        """

        start: float
        end: float
        kind: Memory.Kind
        unit: Memory.Unit

        @property
        def used(self) -> float:
            """The amount of memory used in the interval.

            !!!warning

                This does not track peak memory usage. This will
                only give the difference between the start and end
                of the interval.
            """
            return self.end - self.start

        def to_unit(self, unit: Memory.Unit) -> Memory.Interval:
            """Return the memory used in a different unit.

            Args:
                unit: The unit to convert to.

            Returns:
                The memory used in the new unit.
            """
            if self.unit == unit:
                return self

            return Memory.Interval(
                start=Memory.convert(self.start, self.unit, unit),
                end=Memory.convert(self.end, self.unit, unit),
                kind=self.kind,
                unit=unit,
            )

        @classmethod
        def from_dict(cls, d: Mapping[str, Any]) -> Memory.Interval:
            """Create a memory interval from a dictionary.

            Args:
                d: The dictionary to create from.

            Returns:
                The memory interval.
            """
            return Memory.Interval(
                start=d["start"],
                end=d["end"],
                kind=Memory.Kind.from_str(d["kind"]),
                unit=Memory.Unit.from_str(d["unit"]),
            )

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
                f"{prefix}usage": self.used,
            }

        @classmethod
        def na(cls) -> Memory.Interval:
            """Create a memory interval that represents NA."""
            return Memory.Interval(
                start=0,
                end=0,
                kind=Memory.Kind.NOTSET,
                unit=Memory.Unit.NOTSET,
            )

    class Kind(Enum):
        """An enum for the type of memory."""

        RSS = auto()
        VMS = auto()
        NOTSET = auto()

        def __str__(self) -> str:
            return self.name.lower()

        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, s: str) -> Memory.Kind:
            """Convert a string to a kind."""
            if not isinstance(s, str):
                return Memory.Kind.NOTSET

            _mapping = {
                "rss": Memory.Kind.RSS,
                "vms": Memory.Kind.VMS,
            }
            return _mapping.get(s.lower(), Memory.Kind.NOTSET)

    class Unit(Enum):
        """An enum for the units of time."""

        BYTES = auto()
        KILOBYTES = auto()
        MEGABYTES = auto()
        GIGABYTES = auto()
        NOTSET = auto()

        def __str__(self) -> str:
            return self.name.lower()

        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, s: str) -> Memory.Unit:
            """Convert a string to a unit."""
            if not isinstance(s, str):
                return Memory.Unit.NOTSET

            _mapping = {
                "bytes": Memory.Unit.BYTES,
                "b": Memory.Unit.BYTES,
                "kilobytes": Memory.Unit.KILOBYTES,
                "kb": Memory.Unit.KILOBYTES,
                "megabytes": Memory.Unit.MEGABYTES,
                "mb": Memory.Unit.MEGABYTES,
                "gigabytes": Memory.Unit.GIGABYTES,
                "gb": Memory.Unit.GIGABYTES,
                "notset": Memory.Unit.NOTSET,
            }
            return _mapping.get(s.lower(), Memory.Unit.NOTSET)

    @classmethod
    def convert(
        cls,
        x: float,
        frm: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
        to: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> float:
        """Convert a value from one unit to another.

        Args:
            x: The value to convert.
            frm: The unit of the value.
            to: The unit to convert to.

        Returns:
            The converted value.
        """
        if frm == to:
            return x

        frm = cls.Unit.from_str(frm) if isinstance(frm, str) else frm
        to = cls.Unit.from_str(to) if isinstance(to, str) else to
        return x * _CONVERSION[frm] / _CONVERSION[to]

    @classmethod
    @contextmanager
    def measure(
        cls,
        kind: Memory.Kind | Literal["rss", "vms"] = "rss",
        unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> Iterator[Memory.Interval]:
        """Measure the memory used by a block of code.

        Args:
            kind: The type of memory to measure.
            unit: The unit of memory to use.

        Yields:
            The Memory Interval.
        """
        mem = cls.start(kind=kind, unit=unit)
        interval = mem.stop()
        yield interval
        _interval = mem.stop()
        interval.end = _interval.end

    @classmethod
    def start(
        cls,
        kind: Memory.Kind | Literal["rss", "vms"] = "rss",
        unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> Memory:
        """Start a memory tracker.

        Args:
            kind: The kind of memory to use.
            unit: The unit of memory to use (bytes).

        Returns:
            The Memory tracker.
        """
        return Memory(
            start_memory=Memory.usage(kind=kind, unit=unit),
            kind=kind if isinstance(kind, Memory.Kind) else Memory.Kind.from_str(kind),
            unit=unit if isinstance(unit, Memory.Unit) else Memory.Unit.from_str(unit),
        )

    def stop(self) -> Memory.Interval:
        """Stop the memory tracker.

        Returns:
            The memory interval.
        """
        return Memory.Interval(
            start=self.start_memory,
            end=Memory.usage(kind=self.kind, unit=self.unit),
            kind=self.kind,
            unit=self.unit,
        )

    @classmethod
    def usage(
        cls,
        kind: Memory.Kind | Literal["rss", "vms"] = "rss",
        unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> float:
        """Get the memory used.

        Args:
            kind: The type of memory to measure.
            unit: The unit of memory to use.

        Returns:
            The memory used.
        """
        proc = psutil.Process()
        if kind in (Memory.Kind.RSS, "rss"):
            return Memory.convert(
                proc.memory_info().rss,
                frm=Memory.Unit.BYTES,
                to=unit,
            )

        if kind in (Memory.Kind.VMS, "vms"):
            return Memory.convert(
                proc.memory_info().vms,
                frm=Memory.Unit.BYTES,
                to=unit,
            )

        raise ValueError(f"Unknown memory type: {kind}")


_CONVERSION: Mapping[Memory.Unit, int] = {
    Memory.Unit.BYTES: 1,
    Memory.Unit.KILOBYTES: 1024,
    Memory.Unit.MEGABYTES: 1024**2,
    Memory.Unit.GIGABYTES: 1024**3,
}
