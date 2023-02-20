"""An implementation of an event handler.

This is a simple implementation of an event handler that allows
for the registration and removal of callbacks for events.

It's based on event keys that make to callbacks which
are registered by `on(event_key, callback)` and
can be removed by `remove(event_key, callback)`.

The callbacks are called by `emit(event_key, *args, **kwargs)`
which calls all the handlers for that event.

It's primary use if currently for the `Scheduler`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import logging
import math
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    ParamSpec,
    TypeVar,
    overload,
)

from byop.fluid import ChainPredicate
from byop.functional import funcname
from byop.types import CallbackName

EventKey = TypeVar("EventKey", bound=Hashable)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class Handler(Generic[P, R]):
    """A handler for an event.

    This is a simple class that holds a callback and any predicate
    that must be satisfied for it to be triggered.
    """

    callback: Callable[P, R]
    when: Callable[[], bool] | None = None
    n_called: int = 0
    limit: int | None = None

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R | None:
        """Call the callback if the predicate is satisfied.

        If the predicate is not satisfied, then `None` is returned.
        """
        limit = self.limit if self.limit is not None else math.inf
        if self.n_called >= limit:
            return None

        if self.when is not None and not self.when():
            return None

        self.n_called += 1
        return self.callback(*args, **kwds)


@dataclass
class EventHandler(MutableMapping[CallbackName, Callable[P, R]]):
    """An event."""

    callbacks: dict[CallbackName, Handler[P, R]] = field(default_factory=dict)

    def add(
        self: EventHandler[P, R],
        name: CallbackName,
        callback: Callable[P, R],
        *,
        when: Callable[[], bool] | None = None,
    ) -> None:
        """Add a callback to the event."""
        self.callbacks[name] = Handler(callback, when=when)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> list[R] | None:
        """Emit an event."""
        callbacks = list(self.callbacks.values())
        results = [
            result
            for func in callbacks
            if (result := func(*args, **kwargs)) is not None
        ]
        return results if results else None

    def __iter__(self) -> Iterator[CallbackName]:
        return self.callbacks.__iter__()

    def __len__(self) -> int:
        return self.callbacks.__len__()

    def __getitem__(self, key: CallbackName) -> Callable[P, R]:
        handler = self.callbacks.__getitem__(key)
        return handler.callback

    def __setitem__(
        self: EventHandler[P, R],
        key: CallbackName,
        value: Callable[P, R],
    ) -> None:
        handler = Handler(value)
        self.callbacks.__setitem__(key, handler)

    def __delitem__(self, key: CallbackName) -> None:
        self.callbacks.__delitem__(key)


@dataclass
class EventManager(Mapping[EventKey, EventHandler[Any, R]]):
    """A fairly primitive event handler capable of using an Scheduler."""

    name: str
    handlers: dict[EventKey, EventHandler[Any, R]] = field(
        default_factory=lambda: defaultdict(EventHandler)
    )
    counts: Counter[EventKey] = field(default_factory=Counter)

    @property
    def events(self) -> list[EventKey]:
        """Return a list of the events."""
        return list(self.handlers)

    @overload
    def on(
        self,
        event: EventKey,
        callback: Callable[P, R],
        *,
        every: int | None = ...,
        when: Callable[[], bool] | None = None,
        name: None = None,
    ) -> str:
        ...

    @overload
    def on(
        self,
        event: EventKey,
        callback: Callable[P, R],
        *,
        every: int | None = ...,
        when: Callable[[], bool] | None = None,
        name: CallbackName,
    ) -> CallbackName:
        ...

    def on(
        self,
        event: EventKey,
        callback: Callable[P, R],
        *,
        name: CallbackName | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
    ) -> CallbackName | str:
        """Register a callback for an event."""
        if name is None:
            name = funcname(callback)

        every_pred = None
        if every is not None:
            if every <= 0:
                raise ValueError(f"{every=} must be a positive integer.")
            every_pred = lambda *a, **k: self.counts[event] % every == 0  # noqa: ARG005

        combined_predicate = ChainPredicate() & every_pred & when  # type: ignore
        self.handlers[event].add(name, callback, when=combined_predicate)

        msg = f"{self.name}: Registered callback '{name}' for event {event}"
        if every:
            msg += f" every {every} times"
        if when:
            msg += f" with predicate ({funcname(when)})"
        logger.debug(msg)

        return name

    def emit(
        self,
        event: EventKey,
        *args: Any,
        **kwargs: Any,
    ) -> list[R] | None:
        """Emit an event.

        This will call all the handlers for the event.

        Args:
            event: The event to emit.
            *args: The positional arguments to pass to the handlers.
            **kwargs: The keyword arguments to pass to the handlers.

        Returns:
            A list of the results from the handlers.
        """
        logger.debug(f"{self.name}: Emitting event {event}")
        self.counts[event] += 1

        handler = self.handlers.get(event)
        if handler is None:
            return None

        return handler(*args, **kwargs)

    def __getitem__(self, event: EventKey) -> EventHandler:
        return self.handlers[event]

    def __iter__(self) -> Iterator[EventKey]:
        return iter(self.handlers)

    def __len__(self) -> int:
        return len(self.handlers)

    def has(self, name: CallbackName, *, event: EventKey | None = None) -> bool:
        """Check if the named function exists in the event handlers.

        Args:
            name: The name of the callback to check for.
            event: The event to check for the callback in. If None then
                the callback will be checked for in all events.

        Returns:
            True if the callback exists, False otherwise.
        """
        if event is not None:
            return name in self.handlers[event]

        return any(self.has(name, event=event) for event in self.handlers)

    def remove(self, name: CallbackName, *, event: EventKey | None = None) -> bool:
        """Remove a callback from the event handlers.

        Args:
            name: The name of the callback to remove.
            event: The event to remove the callback from. If None then
                the callback will be removed from all events.

        Returns:
            True if a callback was removed, False otherwise.
        """
        if event is not None:
            handler = self.handlers.get(event)
            if handler is None:
                raise KeyError(f"Event {event} not found.")

            if name in handler:
                logger.debug(f"{self.name}: Removing {name} handler from {event}")
                del handler[name]
                return True

            return False

        truths = [self.remove(name, event=event) for event in self.handlers]
        return any(truths)
