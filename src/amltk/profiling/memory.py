"""Module to measure memory."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import pandas as pd
import psutil

from amltk._functional import dict_get_not_none

if TYPE_CHECKING:
    from pandas._libs.missing import NAType


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
    unit: Memory.Unit | NAType

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Memory.Interval:
        """Create a memory interval from a dictionary.

        Args:
            d: The dictionary to create from.

        Returns:
            The memory interval.
        """
        return Memory.Interval(
            start_vms=dict_get_not_none(d, "start_vms", np.nan),
            start_rss=dict_get_not_none(d, "start_rss", np.nan),
            end_vms=dict_get_not_none(d, "end_vms", np.nan),
            end_rss=dict_get_not_none(d, "end_rss", np.nan),
            unit=Memory.Unit.from_str(dict_get_not_none(d, "unit", pd.NA)),
        )

    @classmethod
    def na(cls) -> Memory.Interval:
        """Create a memory interval that represents NA."""
        return Memory.Interval(
            start_vms=np.nan,
            end_vms=np.nan,
            start_rss=np.nan,
            end_rss=np.nan,
            unit=pd.NA,
        )

    @classmethod
    def convert(
        cls,
        x: float,
        frm: Memory.Unit | NAType | Literal["B", "KB", "MB", "GB"] = "B",
        to: Memory.Unit | NAType | Literal["B", "KB", "MB", "GB"] = "B",
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

        _frm = cls.Unit.from_str(frm) if isinstance(frm, str) else frm
        _to = cls.Unit.from_str(to) if isinstance(to, str) else to
        return x * _CONVERSION[_frm] / _CONVERSION[_to]

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

        try:
            yield interval
        finally:
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
        unit: Memory.Unit | NAType

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

        def to_dict(self, *, prefix: str = "") -> dict[str, Any]:
            """Convert the time interval to a dictionary."""
            return {
                f"{prefix}start_vms": self.start_vms,
                f"{prefix}end_vms": self.end_vms,
                f"{prefix}diff_vms": self.vms_used,
                f"{prefix}start_rss": self.start_rss,
                f"{prefix}end_rss": self.end_rss,
                f"{prefix}diff_rss": self.rss_used,
                f"{prefix}unit": self.unit,
            }

    class Unit(str, Enum):
        """An enum for the units of time."""

        BYTES = "bytes"
        KILOBYTES = "kilobytes"
        MEGABYTES = "megabytes"
        GIGABYTES = "gigabytes"

        @override
        def __str__(self) -> str:
            return self.name.lower()

        @override
        def __repr__(self) -> str:
            return self.name.lower()

        @classmethod
        def from_str(cls, s: Any) -> Memory.Unit | NAType:
            """Convert a string to a unit."""
            if isinstance(s, Memory.Unit):
                return s

            if isinstance(s, str):
                _mapping = {
                    "bytes": Memory.Unit.BYTES,
                    "b": Memory.Unit.BYTES,
                    "kilobytes": Memory.Unit.KILOBYTES,
                    "kb": Memory.Unit.KILOBYTES,
                    "megabytes": Memory.Unit.MEGABYTES,
                    "mb": Memory.Unit.MEGABYTES,
                    "gigabytes": Memory.Unit.GIGABYTES,
                    "gb": Memory.Unit.GIGABYTES,
                }
                return _mapping.get(s.lower(), pd.NA)

            return pd.NA


_CONVERSION: Mapping[Memory.Unit | NAType, float] = {
    Memory.Unit.BYTES: 1,
    Memory.Unit.KILOBYTES: 1024,
    Memory.Unit.MEGABYTES: 1024**2,
    Memory.Unit.GIGABYTES: 1024**3,
    pd.NA: np.nan,
}
