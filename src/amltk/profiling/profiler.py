"""Module to measure memory."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Mapping

from amltk.optimization.trial import mapping_select
from amltk.profiling.memory import Memory
from amltk.profiling.timing import Timer


@dataclass
class Profiler:
    """A profiler for measuring statistics between two events."""

    timer: Timer
    memory: Memory

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Profiler.Interval:
        """Create a profile interval from a dictionary.

        Args:
            d: The dictionary to create from.

        Returns:
            The profile interval.
        """
        return Profiler.Interval(
            memory=Memory.from_dict(mapping_select(d, "memory:")),
            time=Timer.from_dict(mapping_select(d, "time:")),
        )

    @classmethod
    @contextmanager
    def measure(
        cls,
        *,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
        time_kind: Timer.Kind | Literal["wall", "cpu", "process"] = "wall",
    ) -> Iterator[Profiler.Interval]:
        """Profile a block of code.

        !!! note

            * See [`Memory`][amltk.profiling.Memory] for more information on memory.
            * See [`Timer`][amltk.profiling.Timer] for more information on timing.

        Args:
            memory_unit: The unit of memory to use.
            time_kind: The type of timer to use.

        Yields:
            The Profiler Interval. Memory and Timings will not be valid until
            the context manager is exited.
        """
        with Memory.measure(unit=memory_unit) as memory, Timer.time(
            kind=time_kind,
        ) as timer:
            yield Profiler.Interval(memory=memory, time=timer)

    @classmethod
    def start(
        cls,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
        time_kind: Timer.Kind | Literal["wall", "cpu", "process"] = "wall",
    ) -> Profiler:
        """Start a memory tracker.

        !!! note

            * See [`Memory`][amltk.profiling.Memory] for more information on memory.
            * See [`Timer`][amltk.profiling.Timer] for more information on timing.

        Args:
            memory_unit: The unit of memory to use.
            time_kind: The type of timer to use.

        Returns:
            The Memory tracker.
        """
        return Profiler(
            timer=Timer.start(kind=time_kind),
            memory=Memory.start(unit=memory_unit),
        )

    def stop(self) -> Profiler.Interval:
        """Stop the memory tracker.

        Returns:
            The memory interval.
        """
        return Profiler.Interval(
            memory=self.memory.stop(),
            time=self.timer.stop(),
        )

    @classmethod
    def na(cls) -> Profiler.Interval:
        """Create a profile interval that represents NA."""
        return Profiler.Interval(memory=Memory.na(), time=Timer.na())

    @dataclass
    class Interval:
        """A class for representing a profiled interval."""

        memory: Memory.Interval
        time: Timer.Interval

        def to_dict(
            self,
            *,
            prefix: str = "",
            ensure_str: bool = True,
        ) -> dict[str, Any]:
            """Convert the profile interval to a dictionary."""
            return {
                **self.memory.to_dict(prefix=f"{prefix}:", ensure_str=ensure_str),
                **self.time.to_dict(prefix=f"{prefix}:", ensure_str=ensure_str),
            }
