"""Module containing the base protocol of a loader.

For concrete implementations based on the `key` being
a [`Path`][pathlib.Path] see the
[`path_loaders`][amltk.store.paths.path_loaders.PathLoader]
module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")
KeyT_contra = TypeVar("KeyT_contra", contravariant=True)


class Loader(ABC, Generic[KeyT_contra, T]):
    """The base definition of a Loader.

    A Loader is a class that can save and load objects to and from a
    bucket. The Loader is responsible for knowing how to save and load
    objects of a particular type at a given key.
    """

    name: ClassVar[str]
    """The name of the loader."""

    @classmethod
    @abstractmethod
    def can_load(cls, key: KeyT_contra, /, *, check: type[T] | None = None) -> bool:
        """Return True if this loader supports the resource at key.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            key: The key used to identify the resource
            check: If the loader can support loading a specific type
                of object.
        """
        ...

    @classmethod
    @abstractmethod
    def can_save(cls, obj: Any, key: KeyT_contra, /) -> bool:
        """Return True if this loader can save this object.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            obj: The object to save.
            key: The key used to identify the resource
        """
        ...

    @classmethod
    @abstractmethod
    def save(cls, obj: Any, key: KeyT_contra, /) -> None:
        """Save an object to under the given key.

        Args:
            obj: The object to save.
            key: The key to save the object under.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, key: KeyT_contra, /) -> T:
        """Load an object from the given key.

        Args:
            key: The key to load the object from.

        Returns:
            The loaded object.
        """
        ...
