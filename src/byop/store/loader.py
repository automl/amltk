"""Module containing the base protocol of a loader.

For concrete implementations based on the `key` being
a [`Path`][pathlib.Path] see the
[`path_loaders`][byop.store.paths.path_loaders.PathLoader]
module.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

T = TypeVar("T", covariant=True)
KeyT = TypeVar("KeyT", contravariant=True)


class Loader(Protocol[KeyT, T]):
    """The base definition of a Loader.

    A Loader is a class that can save and load objects to and from a
    bucket. The Loader is responsible for knowing how to save and load
    objects of a particular type at a given key.
    """

    @property
    def name(self) -> str:
        """The name of this loader.

        Returns:
            The name of this loader.
        """
        ...

    def can_load(self, key: KeyT, /, *, check: type | None = None) -> bool:
        """Return True if this loader supports the resource at key.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            key: The key used to identify the resource
            check: If the loader can support loading a specific type
                of object.
        """
        ...

    def can_save(self, obj: Any, /) -> bool:
        """Return True if this loader can save this object.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            obj: The key used to identify the resource
        """
        ...

    def save(self, obj: Any, key: KeyT, /) -> None:
        """Save an object to under the given key.

        Args:
            obj: The object to save.
            key: The key to save the object under.
        """
        ...

    def load(self, key: KeyT, /) -> T:
        """Load an object from the given key.

        Args:
            key: The key to load the object from.

        Returns:
            The loaded object.
        """
        ...
