"""A `Drop` in a [`Bucket`][byop.store.bucket.Bucket]
is a reference to a resource.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

from more_itertools.more import first

from byop.functional import funcname

if TYPE_CHECKING:
    from byop.store.loader import Loader


logger = logging.getLogger(__name__)

T = TypeVar("T")
Default = TypeVar("Default")
KeyT = TypeVar("KeyT")


@dataclass
class Drop(Generic[KeyT]):
    """A drop is a reference to a resource in a bucket.

    You likely do not need to create these yourself and
    are a class used by [`Bucket`][byop.store.bucket.Bucket] to wrap
    access to a resource located at a give `key`.

    The main use of this class is to attempt to use different
    [`loaders`][byop.store.loader.Loader] to load a resource
    at a given `key`, using the `key` to try infer which loader
    to use. Each drop has a list of default loaders that it will
    try to use to load the resource.

    For flexibility, you can also specify a `how` when using any
    of [`load`][byop.store.drop.Drop.load], [`get`][byop.store.drop.Drop.get]
    or [`put`][byop.store.drop.Drop.put] to override the default loaders.
    The [`remove`][byop.store.drop.Drop.remove] and
    [`exists`][byop.store.drop.Drop.exists] method also has a `how`
    incase the default methods are not sufficient.

    To support well typed code, you can also specify a `check` type
    which will be used to checked when loading objects, to make sure
    it is of the correct type. This is ignored if `how` is specified.


    The primary methods of interest are
    * [`load`][byop.store.drop.Drop.load]
    * [`get`][byop.store.drop.Drop.get]
    * [`put`][byop.store.drop.Drop.put]
    * [`remove`][byop.store.drop.Drop.remove]
    * [`exists`][byop.store.drop.Drop.exists]

    Args:
        key: The key to the resource.
        loaders: The loaders to use to load the resource.
    """

    __slots__ = ("key", "loaders", "_remove", "_exists")
    key: KeyT
    loaders: tuple[Loader[KeyT, Any], ...]
    _remove: Callable[[KeyT], None]
    _exists: Callable[[KeyT], bool]

    def put(
        self,
        obj: T,
        *,
        how: Callable[[T], None] | None = None,
    ) -> None:
        """Put an object into the bucket.

        Args:
            obj: The object to put into the bucket.
            how: The function to use to put the object into the bucket.
                If `None` then the first loader that can put the object
                will be used.
        """
        if how:
            how(obj)
            return

        loader = first(
            (_l for _l in self.loaders if _l.can_save(obj, self.key)),
            default=None,
        )
        if not loader:
            msg = (
                f"No default way to handle {type(obj)=} objects."
                " Please provide a `how` function that will save"
                f" the object to {self.key}."
            )
            raise ValueError(msg)

        loader.save(obj, self.key)

    @overload
    def load(self, *, check: None = None, how: None = None) -> Any:
        ...

    @overload
    def load(self, *, check: type[T], how: None = None) -> T:
        ...

    @overload
    def load(self, *, check: type[T] | None = ..., how: Callable[[KeyT], T]) -> T:
        ...

    def load(
        self,
        *,
        check: type[T] | None = None,
        how: Callable[[KeyT], T] | None = None,
    ) -> T | Any:
        """Load the resource.

        Args:
            check: By specifying a `type` we check the loaded object of that type, to
                enable correctly typed checked code.
            how: The function to use to load the resource.

        Returns:
            The loaded resource.
        """
        if not isinstance(how, type) and callable(how):
            value = how(self.key)
            loader_name = funcname(how)
        else:
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
    def get(
        self,
        default: None = None,
        *,
        check: None = None,
        how: None = None,
    ) -> Any | None:
        ...

    @overload
    def get(
        self,
        default: Default,
        *,
        check: None = None,
        how: None = None,
    ) -> Default | None:
        ...

    @overload
    def get(
        self,
        default: None = None,
        *,
        check: type[T],
        how: Callable[[KeyT], T] = ...,
    ) -> T:
        ...

    @overload
    def get(
        self,
        default: Default,
        *,
        check: type[T],
        how: Callable[[KeyT], T],
    ) -> Default | T:
        ...

    @overload
    def get(
        self,
        default: Default,
        *,
        check: type[T],
        how: Callable[[KeyT], T] | None = ...,
    ) -> Default | T:
        ...

    def get(
        self,
        default: Default | None = None,
        *,
        check: type[T] | None = None,
        how: Callable[[KeyT], T] | None = None,
    ) -> Default | T | None:
        """Load the resource, or return the default if it can't be loaded.

        See [`load`][byop.store.drop.Drop.load] for more details.

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
            how: The function to use to load the resource.

        Returns:
            The loaded resource or the default value if it cant be loaded.
        """
        try:
            return self.load(check=check, how=how)
        except TypeError as e:
            raise e
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to load {self.key=}: {e}")
            return default

    def remove(self, *, how: Callable[[KeyT], Any] | None = None) -> None:
        """Remove the resource from the bucket.

        Args:
            how: The function to use to remove the resource.
        """
        if how:
            how(self.key)
        else:
            self._remove(self.key)

    def exists(self, *, how: Callable[[KeyT], bool] | None = None) -> bool:
        """Check if the resource exists.

        Returns:
            `True` if the resource exists, `False` otherwise.
        """
        if how:
            return how(self.key)
        return self._exists(self.key)
