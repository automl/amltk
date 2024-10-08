"""The profiler module provides classes for profiling code."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from typing_extensions import override

import pandas as pd

from amltk._functional import mapping_select
from amltk.profiling.memory import Memory
from amltk.profiling.timing import Timer

if TYPE_CHECKING:
    from rich.console import RenderableType

T = TypeVar("T")


@dataclass
class Profile:
    """A profiler for measuring statistics between two events."""

    timer: Timer
    memory: Memory

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Profile.Interval:
        """Create a profile interval from a dictionary.

        Args:
            d: The dictionary to create from.

        Returns:
            The profile interval.
        """
        return Profile.Interval(
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
    ) -> Iterator[Profile.Interval]:
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
            yield Profile.Interval(memory=memory, time=timer)

    @classmethod
    def start(
        cls,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B",
        time_kind: Timer.Kind | Literal["wall", "cpu", "process"] = "wall",
    ) -> Profile:
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
        return Profile(
            timer=Timer.start(kind=time_kind),
            memory=Memory.start(unit=memory_unit),
        )

    def stop(self) -> Profile.Interval:
        """Stop the memory tracker.

        Returns:
            The memory interval.
        """
        return Profile.Interval(
            memory=self.memory.stop(),
            time=self.timer.stop(),
        )

    @classmethod
    def na(cls) -> Profile.Interval:
        """Create a profile interval that represents NA."""
        return Profile.Interval(memory=Memory.na(), time=Timer.na())

    @dataclass
    class Interval:
        """A class for representing a profiled interval."""

        memory: Memory.Interval
        time: Timer.Interval

        def to_dict(self, *, prefix: str = "") -> dict[str, Any]:
            """Convert the profile interval to a dictionary."""
            _prefix = "" if prefix == "" else f"{prefix}:"
            return {
                **self.memory.to_dict(prefix=f"{_prefix}memory:"),
                **self.time.to_dict(prefix=f"{_prefix}time:"),
            }


@dataclass
class Profiler(Mapping[str, Profile.Interval]):
    """Profile and record various events.

    !!! note

        * See [`Memory`][amltk.profiling.Memory] for more information on memory.
        * See [`Timer`][amltk.profiling.Timer] for more information on timing.

    Args:
        memory_unit: The default unit of memory to use.
        time_kind: The default type of timer to use.
    """

    profiles: dict[str, Profile.Interval] = field(default_factory=dict)
    memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] = "B"
    time_kind: Timer.Kind | Literal["wall", "cpu", "process"] = "wall"
    disabled: bool = False
    _running: deque[str] = field(default_factory=deque)

    @override
    def __getitem__(self, key: str) -> Profile.Interval:
        """Get a profile interval."""
        return self.profiles[key]

    @override
    def __iter__(self) -> Iterator[str]:
        """Iterate over the profile names."""
        return iter(self.profiles)

    @override
    def __len__(self) -> int:
        """Get the number of profiles."""
        return len(self.profiles)

    def disable(self) -> None:
        """Disable the profiler."""
        self.disabled = True

    def enable(self) -> None:
        """Enable the profiler."""
        self.disabled = False

    def each(
        self,
        itr: Iterable[T],
        *,
        name: str,
        itr_name: str | Callable[[int, T], str] | None = None,
    ) -> Iterator[T]:
        """Profile each item in an iterable.

        Args:
            itr: The iterable to profile.
            name: The name of the profile that lasts until iteration is complete
            itr_name: The name of the profile for each iteration.
                If a function is provided, it will be called with each item's index
                and the item. It should return a string. If `None` is provided,
                just the index will be used.

        Yields:
            The the items
        """
        match itr_name:
            case None:
                _itr_name = lambda i, _: str(i)
            case str():
                _itr_name = lambda i, _: f"{itr_name}_{i}"
            case _:
                _itr_name = itr_name

        with self.measure(name=name):
            for i, item in enumerate(itr):
                with self.measure(name=_itr_name(i, item)):
                    yield item

    @contextmanager
    def __call__(
        self,
        name: str,
        *,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] | None = None,
        time_kind: Timer.Kind | Literal["wall", "cpu", "process"] | None = None,
    ) -> Iterator[None]:
        """::: amltk.profiling.Profiler.measure"""  # noqa: D415
        with self.measure(name, memory_unit=memory_unit, time_kind=time_kind):
            yield

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] | None = None,
        time_kind: Timer.Kind | Literal["wall", "cpu", "process"] | None = None,
    ) -> Iterator[None]:
        """Profile a block of code. Store the result on this object.

        !!! note

            * See [`Memory`][amltk.profiling.Memory] for more information on memory.
            * See [`Timer`][amltk.profiling.Timer] for more information on timing.

        Args:
            name: The name of the profile.
            memory_unit: The unit of memory to use. Overwrites the default.
            time_kind: The type of timer to use. Overwrites the default.
        """
        if self.disabled:
            yield
            return

        memory_unit = memory_unit or self.memory_unit
        time_kind = time_kind or self.time_kind

        self._running.append(name)
        entry_name = ":".join(self._running)

        with Profile.measure(memory_unit=memory_unit, time_kind=time_kind) as profile:
            self.profiles[entry_name] = profile
            yield

        self._running.pop()

    def df(self) -> pd.DataFrame:
        """Convert the profiler to a dataframe."""
        return pd.DataFrame.from_dict(
            {k: v.to_dict() for k, v in self.profiles.items()},
            orient="index",
        )

    def __rich__(self) -> RenderableType:
        """Render the profiler."""
        from amltk._richutil import df_to_table

        _df = self.df()
        return df_to_table(_df, title="Profiler", index_style="bold")
