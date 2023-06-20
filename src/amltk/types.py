"""Stores low-level types used through the library."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator, List, Mapping, Protocol, Tuple, TypeVar, Union
from typing_extensions import TypeAlias

import numpy as np

Item = TypeVar("Item")
"""The type associated with components, splits and choices"""

Config: TypeAlias = Mapping[str, Any]
"""An object representing a configuration of a pipeline."""

Space = TypeVar("Space")
"""Generic for objects that are aware of a space but not the specific kind"""

Seed: TypeAlias = Union[int, np.random.RandomState, np.random.Generator]
"""Type alias for kinds of Seeded objects"""

FidT = Union[Tuple[int, int], Tuple[float, float], List[Any]]


class Comparable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self: _CT, other: _CT, /) -> bool:
        pass

    @abstractmethod
    def __gt__(self: _CT, other: _CT, /) -> bool:
        pass


_CT = TypeVar("_CT", bound=Comparable)


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
