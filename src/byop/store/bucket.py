"""Module containing the base definition of a bucket.

A bucket is a collection of resources that can be accessed by a key
of a given type. This lets you easily store and retrieve objects of varying
types in a single location.

???+ note "Concrete examples"
    * [`PathBucket`][byop.store.paths.path_bucket.PathBucket].
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    TypeVar,
)

T = TypeVar("T")

if TYPE_CHECKING:
    from byop.store.drop import Drop
    from byop.store.loader import Loader

DEFAULT_FILE_LOADERS: tuple[Loader, ...] = ()

LinkT = TypeVar("LinkT")
KeyT = TypeVar("KeyT", bound=Hashable)


class Bucket(Protocol[LinkT, KeyT]):
    """Definition of a bucket of resources, accessed by a Key.

    Indexing into a bucket returns a [`Drop`][byop.store.drop.Drop] that
    can be used to access the resource.

    The definition mostly follow that of MutableMapping, but with
    the change of `.keys()` and `.values()` to return iterators
    and `.items()` to return an iterator of tuples.
    The other change is that the `.values()` do not return the
    resources themselves, by rather a [`Drop`][byop.store.drop.Drop]
    which wraps the resource.
    """

    def __setitem__(self, key: KeyT, value: Any) -> None:
        """Store a value in the bucket.

        Args:
            key: The key to the resource.
            value: The value to store in the bucket.
        """
        ...

    def __getitem__(self, key: KeyT) -> Drop[LinkT]:
        """Get a drop for a resource in the bucket.

        Args:
            key: The key to the resource.
        """
        ...

    def __delitem__(self, key: KeyT) -> None:
        """Remove a resource from the bucket.

        Args:
            key: The key to the resource.
        """
        ...

    def __iter__(self) -> Iterator[KeyT]:
        """Iterate over the keys in the bucket."""
        ...

    def keys(self) -> Iterator[KeyT]:
        """Iterate over the keys in the bucket."""
        ...

    def values(self) -> Iterator[Drop[LinkT]]:
        """Iterate over the drops in the bucket."""
        ...

    def items(self) -> Iterator[tuple[KeyT, Drop[LinkT]]]:
        """Iterate over the keys and drops in the bucket."""
        ...

    def __contains__(self, key: KeyT) -> bool:
        """Check if a key is in the bucket.

        Args:
            key: The key to check for.
        """
        ...

    def __len__(self) -> int:
        """Get the number of keys in the bucket."""
        ...

    def update(self, items: Mapping[str, Any]) -> None:
        """Update the bucket with the items from a mapping.

        Args:
            items: The items to update the bucket with.
        """
        ...

    def fetch(
        self,
        keys: Iterable[KeyT],
        *,
        default: None | Any | dict[KeyT, Any] = None,
    ) -> dict[KeyT, Any]:
        """Fetch a resource from the bucket.

        Args:
            keys: The keys to the resources.
            default: The default value to return if the key is not in the bucket.
                If a dict is passed, the default for each key will be the value
                in the dict for that key, using None if not present.

        Returns:
            The resources stored in the bucket at the given keys.
        """
        default_dict = {} if not isinstance(default, dict) else default
        return {
            key: self[key].get(default=default_dict.get(key, default)) for key in keys
        }
