"""Stores low-level types used through the library."""
from __future__ import annotations

from typing import Any, Hashable, Iterator, Mapping, TypeVar

from typing_extensions import TypeAlias

# The type associated with components, splits and choices
Item = TypeVar("Item")

# Generic for objects that are aware of a space but not the specific kind
Space = TypeVar("Space", covariant=True)

# Type alias for kinds of Seeded objects
Seed: TypeAlias = int

# The name of an individual step, requires being Hashable
Key = TypeVar("Key", bound=Hashable)

# A name of a pipeline
Name = TypeVar("Name", bound=Hashable)

# A built pipeline object
BuiltPipeline = TypeVar("BuiltPipeline", covariant=True)

# An object representing a configuration of a pipeline.
# Notable examples include a Configuration object from ConfigSpace
# of which Mapping[Key, Any] is a supertype
# TODO: For now we only support Mapping[str, Any] but this serves
# as an abstraction point if we need it.
Config: TypeAlias = Mapping[Key, Any]


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
