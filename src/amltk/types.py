"""Stores low-level types used through the library."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import chain, repeat
from typing import Any, NoReturn, Protocol, TypeAlias, TypeVar
from typing_extensions import override

import numpy as np

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# NOTE: We do not provide sorted types for mutable data structures
# as modifying them could potentially ruin the sorting.

SortedSequence: TypeAlias = Sequence[T]
"""A sequence that is sorted. Only useful for typing"""

SortedIterable: TypeAlias = Iterable[T]
"""An iterable that is sorted. Only useful for typing"""

Item = TypeVar("Item")
"""The type associated with components, splits and choices"""

Config: TypeAlias = Mapping[str, Any]
"""An object representing a configuration of a pipeline."""

Space = TypeVar("Space")
"""Generic for objects that are aware of a space but not the specific kind"""

Seed: TypeAlias = int | np.integer | (np.random.RandomState | np.random.Generator)
"""Type alias for kinds of Seeded objects."""

FidT: TypeAlias = tuple[int, int] | tuple[float, float] | list[Any]
"""Type alias for a fidelity bound."""


class Comparable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self: _CT, other: _CT, /) -> bool:
        pass

    @abstractmethod
    def __gt__(self: _CT, other: _CT, /) -> bool:
        pass


_CT = TypeVar("_CT", bound=Comparable)


def assert_never(value: NoReturn) -> NoReturn:
    """Utility function for asserting that a value is never reached."""
    # This also works in runtime as well:
    raise AssertionError(f"This code should never be reached, got: {value}")


def safe_issubclass(cls: type, classes: str | tuple[str, ...]) -> bool:
    """Check if a class is a subclass of a given type.

    This is a safe version of issubclass that relies on strings,
    which is useful for when the type is not importable.

    Args:
        cls: The class to check
        classes: The type to check for.

    Returns:
        bool
    """

    def type_names(o: type) -> Iterator[str]:
        yield o.__qualname__
        for parent in o.__bases__:
            yield from type_names(parent)

    allowable_names = {classes} if isinstance(classes, str) else set(classes)
    return any(name in allowable_names for name in type_names(cls))


def safe_isinstance(obj: Any, t: str | tuple[str, ...]) -> bool:
    """Check if an object is of a given type.

    This is a safe version of isinstance that relies on strings,
    which is useful for when the type is not importable.

    Args:
        obj: The object to check
        t: The type to check for.

    Returns:
        bool
    """
    return safe_issubclass(type(obj), t)


class Requeue(Iterator[T]):
    """A queue that can have items requeued.

    ```python exec="true" source="material-block" result="python" title="Requeue"
    import random
    from amltk.types import Requeue

    name_generator = iter(["Alice", "Bob", "Charlie"])
    queue: Requeue[str] = Requeue(name_generator)

    rng = random.Random(1)

    def process_name(name: str) -> bool:
        return rng.choice([True, False])

    for name in queue:
        print(f"Processing {name}")
        processed = process_name(name)
        if not processed:
            print(f"Failed to process {name}, requeuing")
            queue.requeue(name)
    ```


    See Also:
        * [`Requeue.from_func(f)`][amltk.types.Requeue.from_func]

            If you have a function which will generate items, you can use
            this to create a requeue from it.

        * [`.append(item)`][amltk.types.Requeue.append]

            Append an item to the end of the queue

        * [`.requeue(item)`][amltk.types.Requeue.requeue]

            Requeue an item to the start of the queue
    """

    def __init__(self, generator: Iterable[T]) -> None:
        """Create a requeue from an iterable."""
        super().__init__()
        self.generator = iter(generator)

    @override
    def __next__(self) -> T:
        return next(self.generator)

    def append(self, item: T) -> None:
        """Append an item to the queue."""
        self.generator = chain(self.generator, [item])

    def requeue(self, item: T) -> None:
        """Requeue an item."""
        self.generator = chain([item], self.generator)

    @classmethod
    def from_func(cls, func: Callable[[], T], n: int | None = None) -> Requeue[T]:
        """Create a Requeue from a function."""
        repeater = repeat(None) if n is None else repeat(None, times=n)
        return cls(func() for _ in repeater)
