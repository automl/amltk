"""Module to measure memory."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any, Iterator, Literal, Mapping

import psutil


@dataclass
class Memory:
    """A timer for measuring the time between two events.

    Attributes:
        start_vms: The virtual memory size at the start of the interval.
        start_rss: The resident set size at the start of the interval.
        unit: The unit of the memory.
    """

    start_vms: float
    start_rss: float
    unit: Memory.Unit

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Memory.Interval:
        """Create a memory interval from a dictionary.

        Args:
            d: The dictionary to create from.

        Returns:
            The memory interval.
        """
        return Memory.Interval(
            start_vms=d["start_vms"],
            start_rss=d["start_rss"],
            end_vms=d["end_vms"],
            end_rss=d["end_rss"],
            unit=Memory.Unit.from_str(d["unit"]),
        )

    @classmethod
    def na(cls) -> Memory.Interval:
        """Create a memory interval that represents NA."""
        return Memory.Interval(
            start_vms=-1,
            end_vms=-1,
            start_rss=-1,
            end_rss=-1,
            unit=Memory.Unit.NOTSET,
        )

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
        unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> Iterator[Memory.Interval]:
        """Measure the memory used by a block of code.

        Args:
            unit: The unit of memory to use.

        Yields:
            The Memory Interval. The start and end memory will not be
            valid until the context manager is exited.

        """
        mem = cls.start(unit=unit)

        interval = Memory.na()
        interval.unit = mem.unit

        yield interval

        _interval = mem.stop()
        interval.start_vms = _interval.start_vms
        interval.end_vms = _interval.end_vms
        interval.start_rss = _interval.start_rss
        interval.end_rss = _interval.end_rss

    @classmethod
    def start(
        cls,
        unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
    ) -> Memory:
        """Start a memory tracker.

        Args:
            unit: The unit of memory to use (bytes).

        Returns:
            The Memory tracker.
        """
        proc = psutil.Process()
        info = proc.memory_info()

        return Memory(
            start_vms=Memory.convert(info.vms, frm=Memory.Unit.BYTES, to=unit),
            start_rss=Memory.convert(info.rss, frm=Memory.Unit.BYTES, to=unit),
            unit=unit if isinstance(unit, Memory.Unit) else Memory.Unit.from_str(unit),
        )

    def stop(self) -> Memory.Interval:
        """Stop the memory tracker.

        Returns:
            The memory interval.
        """
        proc = psutil.Process()
        info = proc.memory_info()

        return Memory.Interval(
            start_vms=self.start_vms,
            start_rss=self.start_rss,
            end_vms=Memory.convert(info.vms, frm=Memory.Unit.BYTES, to=self.unit),
            end_rss=Memory.convert(info.rss, frm=Memory.Unit.BYTES, to=self.unit),
            unit=self.unit,
        )

    @classmethod
    def usage(
        cls,
        kind: Literal["rss", "vms"] = "vms",
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
        _kind = kind.lower()
        if _kind == "rss":
            return Memory.convert(
                proc.memory_info().rss,
                frm=Memory.Unit.BYTES,
                to=unit,
            )

        if _kind == "vms":
            return Memory.convert(
                proc.memory_info().vms,
                frm=Memory.Unit.BYTES,
                to=unit,
            )

        raise ValueError(f"Unknown memory type: {kind}")

    @dataclass
    class Interval:
        """A class for representing a time interval.

        Attributes:
            start_vms: The virtual memory size at the start of the interval.
            start_rss: The resident set size at the start of the interval.
            end_vms: The virtual memory size at the end of the interval.
            end_rss: The resident set size at the end of the interval.
            unit: The unit of memory.
        """

        start_vms: float
        start_rss: float
        end_vms: float
        end_rss: float
        unit: Memory.Unit

        @property
        def vms_used(self) -> float:
            """The amount of vms memory used in the interval.

            !!!warning

                This does not track peak memory usage. This will
                only give the difference between the start and end
                of the interval.
            """
            return self.end_vms - self.start_vms

        @property
        def rss_used(self) -> float:
            """The amount of rss memory used in the interval.

            !!!warning

                This does not track peak memory usage. This will
                only give the difference between the start and end
                of the interval.
            """
            return self.end_rss - self.start_rss

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
                start_vms=Memory.convert(self.start_vms, self.unit, unit),
                end_vms=Memory.convert(self.end_vms, self.unit, unit),
                start_rss=Memory.convert(self.start_rss, self.unit, unit),
                end_rss=Memory.convert(self.end_rss, self.unit, unit),
                unit=unit,
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
                f"{prefix}diff_vms": self.vms_used,
                f"{prefix}diff_rss": self.rss_used,
            }

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


_CONVERSION: Mapping[Memory.Unit, int] = {
    Memory.Unit.BYTES: 1,
    Memory.Unit.KILOBYTES: 1024,
    Memory.Unit.MEGABYTES: 1024**2,
    Memory.Unit.GIGABYTES: 1024**3,
}
