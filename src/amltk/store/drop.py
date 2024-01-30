"""A `Drop` in a [`Bucket`][amltk.store.bucket.Bucket]
is a reference to a resource.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from more_itertools.more import first

from amltk.store.stored import Stored

if TYPE_CHECKING:
    from amltk.store.loader import Loader


logger = logging.getLogger(__name__)

T = TypeVar("T")
Default = TypeVar("Default")
KeyT = TypeVar("KeyT")


@dataclass
class Drop(Generic[KeyT]):
    """A drop is a reference to a resource in a bucket.

    You likely do not need to create these yourself and
    are a class used by [`Bucket`][amltk.store.bucket.Bucket] to wrap
    access to a resource located at a give `key`.

    The main use of this class is to attempt to use different
    [`loaders`][amltk.store.loader.Loader] to load a resource
    at a given `key`, using the `key` to try infer which loader
    to use. Each drop has a list of default loaders that it will
    try to use to load the resource.

    To support well typed code, you can also specify a `check` type
    which will be used to checked when loading objects, to make sure
    it is of the correct type.


    The primary methods of interest are
    * [`load`][amltk.store.drop.Drop.load]
    * [`get`][amltk.store.drop.Drop.get]
    * [`put`][amltk.store.drop.Drop.put]
    * [`remove`][amltk.store.drop.Drop.remove]
    * [`exists`][amltk.store.drop.Drop.exists]
    * [`as_stored`][amltk.store.drop.Drop.as_stored]

    Args:
        key: The key to the resource.
        loaders: The loaders to use to load the resource.
    """

    key: KeyT
    loaders: tuple[type[Loader[KeyT, Any]], ...] = field(repr=False)
    _remove: Callable[[KeyT], bool] = field(repr=False)
    _exists: Callable[[KeyT], bool] = field(repr=False)

    def as_stored(self, read: Callable[[KeyT], T] | None = None) -> Stored[T]:
        """Convert the drop to a [`Stored`][amltk.store.Stored].

        Args:
            read: The method to use to load the resource. If `None` then
                the first loader that can load the resource will be used.

        Returns:
            The drop as a [`Stored`][amltk.store.Stored].
        """
        if read is None:
            loader = first(
                (_l for _l in self.loaders if _l.can_load(self.key)),
                default=None,
            )

            if loader is None:
                raise ValueError(f"Can't load {self.key=} from {self.loaders=}")

            read = loader.load

        return Stored(self.key, read=read)

    def put(self, obj: T) -> Stored[T]:
        """Put an object into the bucket.

        Args:
            obj: The object to put into the bucket.
        """
        loader = first(
            (_l for _l in self.loaders if _l.can_save(obj, self.key)),
            default=None,
        )
        if not loader:
            msg = (
                f"No default way to handle {type(obj)=} objects."
                " Please provide a `Loader` with your `Bucket` to specify"
                f" how to handle this type of object with this extension: {self.key}."
            )
            raise ValueError(msg)

        return loader.save(obj, self.key)

    @overload
    def load(self, *, check: None = None) -> Any:
        ...

    @overload
    def load(self, *, check: type[T]) -> T:
        ...

    def load(self, *, check: type[T] | None = None) -> T | Any:
        """Load the resource.

        Args:
            check: By specifying a `type` we check the loaded object of that type, to
                enable correctly typed checked code.

        Returns:
            The loaded resource.
        """
        loader = first(
            (_l for _l in self.loaders if _l.can_load(self.key)),
            default=None,
        )
        if loader is None:
            raise ValueError(f"Can't load {self.key=} from {self.loaders=}")

        value = loader.load(self.key)
        loader_name = loader.name

        if check is not None and not isinstance(value, check):
            msg = (
                f"Value {value=} loaded by {loader_name=} is not of type {check=},"
                f" but is of type {type(value)=}."
            )
            raise TypeError(msg)

        return value

    @overload
    def get(self, default: None = None, *, check: None = None) -> Any | None:
        ...

    @overload
    def get(self, default: Default, *, check: None = None) -> Any | Default:
        ...

    @overload
    def get(self, default: None = None, *, check: type[T]) -> T | None:
        ...

    @overload
    def get(self, default: Default, *, check: type[T]) -> T | Default:
        ...

    def get(
        self,
        default: Default | None = None,
        *,
        check: type[T] | None = None,
    ) -> Default | T | None:
        """Load the resource, or return the default if it can't be loaded.

        See [`load`][amltk.store.drop.Drop.load] for more details.

        Note:
            This function makes no distinction for the reason it fails to load,
            namely if it's of the incorrect type or the resource does not exist
            at the key.

        Args:
            default: The default value to return if the resource can't be loaded.
            check: By specifying a `type` we check the loaded object of that type, to
                enable correctly typed checked code. If the default value should
                be returned because the resource can't be loaded, then the default
                value is **not** checked.

        Returns:
            The loaded resource or the default value if it cant be loaded.
        """
        try:
            return self.load(check=check)
        except TypeError as e:
            raise e
        except FileNotFoundError:
            return default
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to load {self.key=} from {self.loaders=}: {e}",
                exc_info=True,
            )

        return None

    def remove(self) -> bool:
        """Remove the resource from the bucket.

        !!! note "Non-existent resources"

            If the resource does not exist, then the function will return `True`.
        """
        return self._remove(self.key)

    def exists(self) -> bool:
        """Check if the resource exists.

        Returns:
            `True` if the resource exists, `False` otherwise.
        """
        return self._exists(self.key)
