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
import logging
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    ParamSpec,
    TypeVar,
)
from uuid import uuid4

from attrs import field, frozen

EventKey = TypeVar("EventKey", bound=Hashable)

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@frozen
class Handler(Generic[P, R]):
    """A handler for an event.

    This is a simple class that holds a callback and any predicate
    that must be satisfied for it to be triggered.
    """

    callback: Callable[P, R | None]
    pred: Callable[P, bool] | None = None

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R | None:
        """Call the callback if the predicate is satisfied.

        If the predicate is not satisfied, then `None` is returned.
        """
        if self.pred and not self.pred(*args, **kwds):
            return None

        return self.callback(*args, **kwds)


@frozen
class EventHandler(MutableMapping[str, Callable[P, Optional[R]]]):
    """An event."""

    callbacks: dict[str, Handler[P, R]] = field(factory=dict)

    def add(
        self: EventHandler[P, R],
        name: str,
        callback: Callable[P, R | None],
        *,
        pred: Callable[P, bool] | None = None,
    ) -> None:
        """Add a callback to the event."""
        self.callbacks[name] = Handler(callback, pred)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> list[R] | None:
        """Emit an event."""
        results = [
            result
            for func in self.callbacks.values()
            if (result := func(*args, **kwargs)) is not None
        ]
        return results if results else None

    def __iter__(self) -> Iterator[str]:
        return self.callbacks.__iter__()

    def __len__(self) -> int:
        return self.callbacks.__len__()

    def __getitem__(self, key: str) -> Callable[P, R | None]:
        handler = self.callbacks.__getitem__(key)
        return handler.callback

    def __setitem__(
        self: EventHandler[P, R],
        key: str,
        value: Callable[P, R | None],
    ) -> None:
        handler = Handler(value)
        self.callbacks.__setitem__(key, handler)

    def __delitem__(self, key: str) -> None:
        self.callbacks.__delitem__(key)


@frozen
class EventManager(Mapping[EventKey, EventHandler[Any, R]]):
    """A fairly primitive event handler capable of using an Scheduler."""

    name: str
    handlers: dict[EventKey, EventHandler[Any, R]] = field(
        factory=lambda: defaultdict(EventHandler)
    )
    count: Counter[EventKey] = field(factory=lambda: Counter())

    @property
    def events(self) -> list[EventKey]:
        """Return a list of the events."""
        return list(self.handlers)

    def __call__(self, event: EventKey, handler: Callable[..., R | None]) -> None:
        """Register a handler for an event."""
        self.on(event, handler)

    def on(
        self,
        event: EventKey,
        callback: Callable[P, R | None],
        pred: Callable[P, bool] | None = None,
        *,
        name: str | None = None,
    ) -> str:
        """Register a callback for an event."""
        if name is None:
            name = str(uuid4())

        self.handlers[event].add(name, callback, pred=pred)

        msg = f"{self.name}: Registered {name} ({callback}) for event {event}"
        if pred:
            msg += f" with predicate ({pred})"
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
        self.count[event] += 1

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

    def has(self, name: str, *, event: EventKey | None = None) -> bool:
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

    def remove(self, name: str, *, event: EventKey | None = None) -> bool:
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
