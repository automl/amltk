"""All code for allowing an event system."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
import logging
import math
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    ParamSpec,
    TypeVar,
    overload,
)

from byop.fluid import ChainPredicate
from byop.functional import callstring, funcname

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event(Generic[P]):
    """An event that can be emitted."""

    name: str

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class SubcriberDecorator(Generic[P]):
    """A decorator that allows subscribing to an event when called.

    Attributes:
        manager: The event manager to use.
        event: The event to subscribe to.
        name: The name of the event.
        when: A predicate that must be satisfied for the event to be triggered.
        limit: The maximum number of times the event can be triggered.
        repeat: The number of times the event has been triggered.
        every: The number of times the event must be triggered before
            the callback is called.
    """

    manager: EventManager
    event: Event[P] | Hashable
    name: str | None = None
    when: Callable[[], bool] | None = None
    limit: int | None = None
    repeat: int = 1
    every: int | None = None

    def __call__(self, callback: Callable[P, Any]) -> Callable[P, Any]:
        """Decorate a function to subscribe to an event."""
        self.manager.on(
            self.event,
            callback,
            name=self.name,
            when=self.when,
            limit=self.limit,
            repeat=self.repeat,
            every=self.every,
        )
        return callback


@dataclass
class Subscriber(Generic[P]):
    """An attachable object that allows subscribing to an event when called.

    Attributes:
        manager: The event manager to use.
        event: The event to subscribe to.
    """

    manager: EventManager
    event: Event[P] | Hashable

    @overload
    def __call__(
        self,
        callback: None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> SubcriberDecorator[P]:
        ...

    @overload
    def __call__(
        self,
        callback: Callable[P, Any],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> Callable[P, Any]:
        ...

    def __call__(
        self,
        callback: Callable[P, Any] | None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> Callable[P, Any] | SubcriberDecorator[P]:
        """Subscribe to an event."""
        if callback is None:
            return SubcriberDecorator(
                manager=self.manager,
                event=self.event,
                name=name,
                when=when,
                limit=limit,
                repeat=repeat,
                every=every,
            )

        self.manager.on(
            self.event,
            callback,
            name=name,
            when=when,
            limit=limit,
            repeat=repeat,
            every=every,
        )
        return callback


@dataclass
class Handler(Generic[P]):
    """A handler for an event.

    This is a simple class that holds a callback and any predicate
    that must be satisfied for it to be triggered.
    """

    callback: Callable[P, Any]
    when: Callable[[], bool] | None = None
    n_called: int = 0
    limit: int | None = None
    repeat: int = 1

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Call the callback if the predicate is satisfied.

        If the predicate is not satisfied, then `None` is returned.
        """
        limit = self.limit if self.limit is not None else math.inf
        if self.n_called >= limit:
            return

        if self.when is not None and not self.when():
            return

        self.n_called += 1
        logger.debug(f"Calling: {callstring(self.callback, *args, **kwargs)}")
        self.callback(*args, **kwargs)


@dataclass
class EventHandler(Mapping[str, list[Callable[P, Any]]]):
    """An event handler."""

    callbacks: dict[str, list[Handler[P]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(
        self: EventHandler[P],
        name: str,
        callback: Callable[P, Any],
        *,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
    ) -> None:
        """Add a callback to the event."""
        handler = Handler(callback, when=when, repeat=repeat, limit=limit)
        self.callbacks[name].append(handler)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Emit an event."""
        for handler in chain.from_iterable(self.callbacks.values()):
            for _ in range(handler.repeat):
                handler(*args, **kwargs)

    def __iter__(self) -> Iterator[str]:
        return self.callbacks.__iter__()

    def __len__(self) -> int:
        return self.callbacks.__len__()

    def __getitem__(self, key: str) -> list[Callable[P, Any]]:
        handlers = self.callbacks.__getitem__(key)
        return [handler.callback for handler in handlers]

    def __delitem__(self, key: str) -> None:
        self.callbacks.__delitem__(key)


class EventManager(Mapping[Hashable, EventHandler[Any]]):
    """A fairly primitive event handler capable of using an Scheduler.

    It's based on event keys that make to callbacks which
    are registered by `on(event_key, callback)` and
    can be removed by `remove(event_key, callback)`.

    The callbacks are called by `emit(event_key, *args, **kwargs)`
    which calls all the handlers for that event.

    Attributes:
        name: The name of the event manager.
        handlers: A mapping of event keys to handlers.
        counts: A counter of the number of times each event has been
            emitted.
        forwards: A mapping of event keys to event keys. When an event
            is emitted, the event is forwarded to the event key
            specified in this mapping.
    """

    def __init__(self, name: str) -> None:
        """Initialize the event manager.

        Args:
            name: The name of the event manager.
        """
        self.name = name
        self.handlers: dict[Hashable, EventHandler[Any]] = defaultdict(EventHandler)
        self.counts: Counter[Hashable] = Counter()
        self.forwards: dict[Hashable, list[Hashable]] = defaultdict(list)

    @property
    def events(self) -> list[Hashable]:
        """Return a list of the events."""
        return list(self.handlers)

    def on(
        self,
        event: Hashable,
        callback: Callable,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> str:
        """Register a callback for an event.


        Args:
            event: The event to register the callback for.
            callback: The callback to register.
            name: The name of the callback. If not provided, then the
                name of the callback is used.
            when: A predicate that must be satisfied for the callback
                to be called.
            every: The callback will be called every `every` times
                the event is emitted.
            repeat: The callback will be called `repeat` times
                successively.
            limit: The maximum number of times the callback can be
                called.
        """
        if repeat <= 0:
            raise ValueError(f"{repeat=} must be a positive integer.")

        if name is None:
            name = funcname(callback)

        every_pred = None
        if every is not None:
            if every <= 0:
                raise ValueError(f"{every=} must be a positive integer.")
            every_pred = lambda *a, **k: self.counts[event] % every == 0  # noqa: ARG005

        combined_predicate = ChainPredicate() & every_pred & when  # type: ignore
        self.handlers[event].add(
            name,
            callback,
            when=combined_predicate,
            repeat=repeat,
            limit=limit,
        )

        msg = f"{self.name}: Registered callback '{name}' for event {event}"
        if every:
            msg += f" every {every} times"
        if when:
            msg += f" with predicate ({funcname(when)})"
        if repeat > 1:
            msg += f" called {repeat} times successively"
        logger.debug(msg)

        return name

    def emit(self, event: Hashable, *args: Any, **kwargs: Any) -> None:
        """Emit an event.

        This will call all the handlers for the event.

        Args:
            event: The event to emit.
            *args: The positional arguments to pass to the handlers.
            **kwargs: The keyword arguments to pass to the handlers.

        Returns:
            A list of the results from the handlers.
        """
        logger.debug(f"{self.name}: Emitting {event} with {args=} and {kwargs=}")
        self.counts[event] += 1

        handler = self.handlers.get(event)
        if handler is None:
            return

        handler(*args, **kwargs)
        if event in self.forwards:
            fwds: list[Hashable] = self.forwards[event]
            for fwd in fwds:
                logger.debug(f"Forwarding {event} to {fwd}")
                self.emit(fwd, *args, **kwargs)

    def __getitem__(self, event: Hashable) -> EventHandler:
        return self.handlers[event]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.handlers)

    def __len__(self) -> int:
        return len(self.handlers)

    def has(self, name: str, *, event: Hashable | None = None) -> bool:
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

    def remove(self, name: str, *, event: Hashable | None = None) -> bool:
        """Remove any callback(s) registered with `name` from the event handlers.

        Args:
            name: The name of the callback(s) to remove.
            event: The event to remove the callback(s) from. If `None` then
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

    def forward(self, frm: Hashable, to: Hashable) -> None:
        """Forward an event to another event.

        Args:
            frm: The event to forward.
            to: The event to forward to.
        """
        self.forwards[frm].append(to)

    @overload
    def subscriber(self, event: Event[P]) -> Subscriber[P]:
        ...

    @overload
    def subscriber(self, event: tuple[Event[P], Hashable]) -> Subscriber[P]:
        ...

    @overload
    def subscriber(self, event: Hashable) -> Subscriber[Any]:
        ...

    def subscriber(
        self,
        event: Event[P] | tuple[Event[P], Hashable] | Hashable,
    ) -> Subscriber[P]:
        """Create a subscriber for an event.

        Args:
            event: The event that will be subscribed to

        Returns:
            A subscriber for the event.
        """
        return Subscriber(self, event)
