"""All code for allowing an event system."""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec, override
from uuid import uuid4

from amltk.fluid import ChainPredicate
from amltk.functional import callstring, funcname

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTimeCallOrderStrategy:
    """A calling strategy that calls callbacks in the order they were registered."""

    @classmethod
    def execute(
        cls,
        events: Iterable[
            tuple[EventHandler[P], tuple[Any, ...] | None, dict[str, Any] | None]
        ],
    ) -> None:
        """Call all events in the scheduler."""
        handlers = []
        for event_handler, args, kwargs in events:
            handlers += [
                (handler, args or (), kwargs or {})
                for handler in event_handler.handlers
            ]

        sorted_handlers = sorted(handlers, key=lambda item: item[0].registered_at)
        for handler, args, kwargs in sorted_handlers:
            handler(*args, **kwargs)


@dataclass(frozen=True)
class Event(Generic[P]):
    """An event that can be emitted.

    Attributes:
        name: The name of the event.
    """

    name: str


@dataclass
class Subscriber(Generic[P]):
    """An object that can be used to easily subscribe to a certain event.

    ```python
    from amltk.events import Event, EventManager, Subscriber

    test_event: Event[[int, str]] = Event("test")

    emitter = Emitter("hello world")
    subscribe = emitter.subscriber(test_event)

    # Use it as a decorator

    @subscribe
    def callback(a: int, b: str) -> None:
        print(f"Got {a} and {b}!")

    # ... or just pass a function

    subscribe(callback)

    # Will emit `test_event` with the arguments 1 and "hello"
    # and call the callback with those same arguments.
    emitter.emit(test_event, 1, "hello")
    ```

    Attributes:
        manager: The event manager to use.
        event: The event to subscribe to.
    """

    emitter: Emitter
    event: Event[P]
    name: str | None = None
    when: Callable[[], bool] | None = None
    limit: int | None = None
    repeat: int = 1
    every: int | None = None

    @property
    def event_counts(self) -> int:
        """The number of times this event has been emitted."""
        return self.emitter.event_counts[self.event]

    @overload
    def __call__(
        self,
        callback: None = None,
        *,
        name: str | None = ...,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int | None = ...,
    ) -> partial[Callable[P, Any]]:
        ...

    @overload
    def __call__(
        self,
        callback: Callable[P, Any] | Iterable[Callable[P, Any]],
        *,
        name: str | None = ...,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int | None = ...,
    ) -> None:
        ...

    def __call__(
        self,
        callback: Callable[P, Any] | Iterable[Callable[P, Any]] | None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> None | partial[Callable[P, Any]]:
        """Subscribe to the event associated with this object.

        Args:
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

        Returns:
            The callback if it was provided, otherwise it acts
                as a decorator.
        """
        if callback is None:
            return partial(
                self.__call__,
                name=name,
                when=when,
                limit=limit,
                repeat=repeat,
                every=every,
            )  # type: ignore

        self.emitter.on(
            self.event,
            callback,
            name=name,
            when=when,
            limit=limit,
            repeat=repeat,
            every=every,
        )
        return None

    def emit(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Emit this subscribers event."""
        self.emitter.emit(self.event, *args, **kwargs)


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
    registered_at: int = field(default_factory=time.time_ns)

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
        logger.debug(f"Calling: {callstring(self.callback)}")
        self.callback(*args, **kwargs)


@dataclass
class EventHandler(Mapping[str, List[Callable[P, Any]]]):
    """An event handler."""

    callbacks: dict[str, list[Handler[P]]] = field(
        default_factory=lambda: defaultdict(list),
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

    @override
    def __iter__(self) -> Iterator[str]:
        return self.callbacks.__iter__()

    @override
    def __len__(self) -> int:
        return self.callbacks.__len__()

    @override
    def __getitem__(self, key: str) -> list[Callable[P, Any]]:
        handlers = self.callbacks.__getitem__(key)
        return [handler.callback for handler in handlers]

    def __delitem__(self, key: str) -> None:
        self.callbacks.__delitem__(key)

    @property
    def handlers(self) -> Iterable[Handler[P]]:
        """Return the handlers sorted by registration time."""
        return chain.from_iterable(self.callbacks.values())  # type: ignore


class Emitter(Mapping[Event, EventHandler]):
    """An event emitter.

    This is a convenience class that wraps an event manager and provides
    a way to emit events. The events emitter and subscribed to will be
    identified by a UUID, such that two objects emitting the same event
    will have a different set of listeners who will be called. For
    downstream users, this means they must subscribe to events directly
    from the object they are using.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialise the emitter.

        Args:
            name: The name of the emitter. If not provided, then a UUID
                will be used.
        """
        super().__init__()
        self.unique_ref = f"{name}-{uuid4()}"
        self.emitted_events: set[Event] = set()

        self.name: str = name if name is not None else self.unique_ref
        self.handlers: dict[Event, EventHandler[Any]] = defaultdict(EventHandler)
        self.event_counts: Counter[Event] = Counter()

    def emit(self, event: Event[P], *args: Any, **kwargs: Any) -> None:
        """Emit an event.

        This will call all the handlers for the event.

        Args:
            event: The event to emit.
                If passing a list, then the handlers for all events will be called,
                regardless of the order
            *args: The positional arguments to pass to the handlers.
            **kwargs: The keyword arguments to pass to the handlers.

        Returns:
            A list of the results from the handlers.
        """
        logger.debug(f"{self.name}: Emitting {event}")

        self.event_counts[event] += 1
        if handler := self.handlers.get(event):
            handler(*args, **kwargs)

    def emit_many(
        self,
        events: dict[Event, tuple[tuple[Any, ...] | None, dict[str, Any] | None]],
    ) -> None:
        """Emit multiple events.

        This is useful for cases where you don't want to favour one callback
        over another, and so uses the time a callback was registered to call
        the callback instead.

        Args:
            events: A mapping of event keys to a tuple of positional
                arguments and keyword arguments to pass to the handlers.
        """
        for event in events:
            self.event_counts[event] += 1

        items = [
            (event_handler, args, kwargs)
            for name, (args, kwargs) in events.items()
            if (event_handler := self.get(name)) is not None
        ]

        header = f"{self.name}: Emitting many events"
        logger.debug(header)
        logger.debug(",".join(str(event) for event in events))
        RegisteredTimeCallOrderStrategy.execute(items)

    @override
    def __getitem__(self, event: Event) -> EventHandler:
        return self.handlers[event]

    @override
    def __iter__(self) -> Iterator[Event]:
        return iter(self.handlers)

    @override
    def __len__(self) -> int:
        return len(self.handlers)

    @property
    def events(self) -> list[Event]:
        """Return a list of the events."""
        return list(self.handlers.keys())

    def subscriber(
        self,
        event: Event[P],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> Subscriber[P]:
        """Create a subscriber for an event.

        Args:
            event: The event to register the callback for.
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

        Returns:
            A subscriber for the event.
        """
        return Subscriber(
            self,
            event,
            name=name,
            when=when,
            every=every,
            repeat=repeat,
            limit=limit,
        )

    def on(
        self,
        event: Event[P],
        callback: Callable | Iterable[Callable],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> None:
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

        if every is not None and every > 0:
            raise ValueError(f"{every=} must be a positive integer.")

        every_pred = None
        if every is not None and every <= 0:
            assert isinstance(event, Event)
            every_pred = lambda *_, **__: self.event_counts[event] % every == 0

        combined_predicate = ChainPredicate() & every_pred & when  # type: ignore

        # This hackery is just to get down to a flat list of events that need
        # to be set up
        callbacks = [callback] if callable(callback) else list(callback)
        for c in callbacks:
            _name = funcname(c) if name is None else name
            self.handlers[event].add(
                _name,
                c,
                when=combined_predicate,
                repeat=repeat,
                limit=limit,
            )

            msg = f"{self.name}: Registered callback '{_name}' for event {event}"
            if every:
                msg += f" every {every} times"
            if when:
                msg += f" with predicate ({funcname(when)})"
            if repeat > 1:
                msg += f" called {repeat} times successively"
            logger.debug(msg)
