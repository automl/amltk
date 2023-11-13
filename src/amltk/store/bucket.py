"""Module containing the base definition of a bucket.

A bucket is a collection of resources that can be accessed by a key
of a given type. This lets you easily store and retrieve objects of varying
types in a single location.

???+ note "Concrete examples"

    * [`PathBucket`][amltk.store.paths.path_bucket.PathBucket].
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload
from typing_extensions import override

from more_itertools import ilen

from amltk.store.drop import Drop

T = TypeVar("T")

if TYPE_CHECKING:
    from typing_extensions import Self

    from amltk.store.loader import Loader

DEFAULT_FILE_LOADERS: tuple[Loader, ...] = ()

LinkT = TypeVar("LinkT")
KeyT = TypeVar("KeyT", bound=Hashable)


class Bucket(ABC, MutableMapping[KeyT, Drop[LinkT]], Generic[KeyT, LinkT]):
    """Definition of a bucket of resources, accessed by a Key.

    Indexing into a bucket returns a [`Drop`][amltk.store.drop.Drop] that
    can be used to access the resource.

    The definition mostly follow that of MutableMapping, but with
    the change of `.keys()` and `.values()` to return iterators
    and `.items()` to return an iterator of tuples.
    The other change is that the `.values()` do not return the
    resources themselves, by rather a [`Drop`][amltk.store.drop.Drop]
    which wraps the resource.
    """

    @override
    @abstractmethod
    def __setitem__(self, key: KeyT, value: Any) -> None:
        """Store a value in the bucket.

        Args:
            key: The key to the resource.
            value: The value to store in the bucket.
        """

    @override
    @abstractmethod
    def __getitem__(self, key: KeyT) -> Drop[LinkT]:
        """Get a drop for a resource in the bucket.

        Args:
            key: The key to the resource.
        """

    @override
    @abstractmethod
    def __delitem__(self, key: KeyT) -> None:
        """Remove a resource from the bucket.

        Args:
            key: The key to the resource.
        """

    @override
    @abstractmethod
    def __iter__(self) -> Iterator[KeyT]:
        """Iterate over the keys in the bucket."""

    @abstractmethod
    def sub(self, key: KeyT) -> Self:
        """Create a subbucket of this bucket.

        Args:
            key: The name of the sub bucket.

        Returns:
            A new bucket with the same loaders as the current bucket.
        """

    @override
    def __contains__(self, key: object) -> bool:
        """Check if a key is in the bucket.

        Args:
            key: The key to check for.
        """
        return key in self

    @override
    def __len__(self) -> int:
        """Get the number of keys in the bucket."""
        return ilen(iter(self))

    @overload
    def find(self, pattern: str) -> dict[str, Drop[LinkT]] | None:
        ...

    @overload
    def find(
        self,
        pattern: str,
        *,
        multi_key: Literal[False] = False,
    ) -> dict[str, Drop[LinkT]] | None:
        ...

    @overload
    def find(
        self,
        pattern: str,
        *,
        multi_key: Literal[True],
    ) -> dict[tuple[str, ...], Drop[LinkT]] | None:
        ...

    @overload
    def find(
        self,
        pattern: str,
        *,
        multi_key: bool,
    ) -> dict[str, Drop[LinkT]] | dict[tuple[str, ...], Drop[LinkT]] | None:
        ...

    def find(
        self,
        pattern: str,
        *,
        multi_key: bool = False,
    ) -> dict[str, Drop[LinkT]] | dict[tuple[str, ...], Drop[LinkT]] | None:
        """Find resources in the bucket.

        ```python
        found = bucket.find(r"trial_(.+)_val_predictions.npy")  # (1)!
        if found is None:
            raise KeyError("No predictions found")

        for name, drop in found.items():
            predictions = drop.get()
            # Do something with the predictions
            # ...
        ```

        1. The `(.+)` is a **capture group** which will attempt to match anything `.`,
            when there is one or more occurences `+`, and put it in a capure group `()`.
            What is captured will be used as the key in the returned dict.

        Args:
            pattern: The pattern to search for.
            multi_key: Whether you have multiple capture groups in the pattern.

                !!! note "Multiple capture groups with `()`"

                    If using multiple capture groups, the returned dict will have
                    tuples as keys. If there is only one capture group, the tuple
                    will be expanded to a single value.

        Returns:
            A mapping of links to drops for the resources found.
        """
        keys = [(key, match) for key in self if (match := re.search(pattern, str(key)))]
        if not keys:
            return None

        matches = {match.groups(): self[key] for key, match in keys}

        # If it's a tuple of length 1, we expand it
        one_group = len(next(iter(matches.keys()))) == 1
        if one_group:
            if multi_key:
                raise ValueError(
                    "Use multi_key=True when the pattern has more than 1 capture group",
                )

            return {key[0]: drop for key, drop in matches.items()}

        # Here we have multi-groups => tuple keys
        if not multi_key:
            raise ValueError(
                "Use multi_key=False when the pattern has only 1 capture group",
            )

        return matches

    def store(self, other: Mapping[KeyT, Any]) -> None:
        """Store items into the bucket with the given mapping.

        Args:
            other: The mapping of items to store in the bucket.
        """
        for key, value in other.items():
            self[key] = value

    def fetch(
        self,
        *keys: KeyT,
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

    @override
    def update(self, items: Mapping[KeyT, Any]) -> None:  # type: ignore
        """Update the bucket with the given mapping.

        Args:
            items: The mapping of items to store in the bucket.
        """
        for key, value in items.items():
            self[key].put(value)

    def remove(
        self,
        keys: Iterable[KeyT],
        *,
        how: Callable[[LinkT], bool] | None = None,
    ) -> dict[KeyT, bool]:
        """Remove resources from the bucket.

        Args:
            keys: The keys to the resources.
            how: A function that removes the resource.

        Returns:
            A mapping of keys to whether they were removed.
        """
        return {key: self[key].remove(how=how) for key in keys}

    def __truediv__(self, key: KeyT) -> Self:
        try:
            return self.sub(key)
        except TypeError:
            return NotImplemented
