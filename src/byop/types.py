"""Stores low-level types used through the library."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Hashable, Iterator, ParamSpec, Protocol, TypeVar, Union

import numpy as np
from typing_extensions import TypeAlias

Item = TypeVar("Item")
"""The type associated with components, splits and choices"""

Space = TypeVar("Space")
"""Generic for objects that are aware of a space but not the specific kind"""

Seed: TypeAlias = Union[int, np.random.RandomState, np.random.Generator]
"""Type alias for kinds of Seeded objects"""

Key = TypeVar("Key", bound=Hashable)
"""The name of an individual step, requires being Hashable"""

Name = TypeVar("Name", bound=Hashable)
"""A name of a pipeline"""

ResultKey = TypeVar("ResultKey", bound=Hashable)
"""The key for a result"""

BuiltPipeline = TypeVar("BuiltPipeline", covariant=True)
"""A built pipeline object"""

TaskParams = ParamSpec("TaskParams")
"""The paramspec of a task"""

TaskReturn = TypeVar("TaskReturn")
"""The return type of a task"""

TrialResult = TypeVar("TrialResult")
"""Something you tell the optimizer about"""
TrialResult_co = TypeVar("TrialResult_co")
TrialResult_contra = TypeVar("TrialResult_contra")

TaskName: TypeAlias = Hashable
"""A name for a task"""

CallbackName: TypeAlias = Hashable
"""A name for a callback"""

Msg: TypeAlias = Any
"""A message or response to/from a comm task"""

Config = TypeVar("Config")
"""An object representing a configuration of a pipeline."""
Config_co = TypeVar("Config_co")
Config_contra = TypeVar("Config_contra")


class Comparable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self: CT, other: CT, /) -> bool:
        pass

    @abstractmethod
    def __gt__(self: CT, other: CT, /) -> bool:
        pass


CT = TypeVar("CT", bound=Comparable)


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
