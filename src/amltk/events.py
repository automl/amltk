"""All code for allowing an event system."""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain, product
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
    event: Event[P] | tuple[Event[P], ...]
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
    """An object that can be used to easily subscribe to a certain event.

    ```python
    from amltk.events import Event, EventManager, Subscriber

    event_manager = EventManager()
    test_event: Event[[int, str]] = Event("test")

    subscribe_to_test_event = Subscriber(event_manager, test_event)

    # Use it as a decorator

    @subscribe_to_test_event
    def callback(a: int, b: str) -> None:
        print(f"Got {a} and {b}!")

    # ... or just pass a function

    subscribe_to_test_event(callback)

    # Will emit `test_event` with the arguments 1 and "hello"
    # and call the callback with those same arguments.
    event_manager.emit(test_event, 1, "hello")
    ```

    Attributes:
        manager: The event manager to use.
        event: The event to subscribe to.
    """

    manager: EventManager
    event: Event[P] | tuple[Event[P], ...]

    @property
    def count(self) -> int:
        """The number of times this event has been emitted."""
        if isinstance(self.event, Event):
            return self.manager.counts[self.event]

        return sum(self.manager.counts[event] for event in self.event)

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
        callback: Callable[P, Any] | Iterable[Callable[P, Any]],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
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
    ) -> None | SubcriberDecorator[P]:
        """Subscribe to this subscriberes event.

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
        return None

    def forward(self, to: Subscriber[P]) -> None:
        """Forward events to another subscriber.

        Args:
            to: The subscriber to forward events to.
        """
        forward_function = to.emit
        self(forward_function)

    def remove(self, callback_or_name: Callable | str) -> bool:
        """Remove a callback from this subscribers event.

        Args:
            callback_or_name: The callback to remove or the name
                of the callback to remove.

        Returns:
            Whether a callback was removed or not.
        """
        if callable(callback_or_name):
            name = funcname(callback_or_name)
        else:
            name = callback_or_name

        return self.manager.remove(name, event=self.event)

    def emit(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Emit this subscribers event."""
        self.manager.emit(self.event, *args, **kwargs)


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


class EventManager(Mapping[Event, EventHandler[Any]]):
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
        super().__init__()
        self.name: str = name
        self.handlers: dict[Event, EventHandler[Any]] = defaultdict(EventHandler)
        self.counts: Counter[Event] = Counter()
        self.forwards: dict[Event, list[Event]] = defaultdict(list)

    @property
    def events(self) -> list[Event]:
        """Return a list of the events."""
        return list(self.handlers)

    def on(
        self,
        event: Event[P] | Iterable[Event[P]],
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

        if every is not None and not isinstance(event, Event):
            # TODO: We could technically support this
            raise NotImplementedError(
                f"Cannot use {every=} with multiple events. "
                "Use a predicate instead.",
            )

        every_pred = None
        if every is not None and every <= 0:
            assert isinstance(event, Event)
            every_pred = lambda *a, **k: self.counts[event] % every == 0  # noqa: ARG005

        combined_predicate = ChainPredicate() & every_pred & when  # type: ignore

        # This hackery is just to get down to a flat list of events that need
        # to be set up
        all_events: list[Event] = [event] if isinstance(event, Event) else list(event)

        callbacks = [callback] if callable(callback) else list(callback)
        for e, _callback in product(all_events, callbacks):
            self.handlers[e].add(
                funcname(_callback) if name is None else name,
                _callback,
                when=combined_predicate,
                repeat=repeat,
                limit=limit,
            )

            msg = f"{self.name}: Registered callback '{name}' for event {e}"
            if every:
                msg += f" every {every} times"
            if when:
                msg += f" with predicate ({funcname(when)})"
            if repeat > 1:
                msg += f" called {repeat} times successively"
            logger.debug(msg)

    def emit(
        self,
        event: Event[P] | tuple[Event[P], ...],
        *args: Any,
        **kwargs: Any,
    ) -> None:
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

        _events = [event] if isinstance(event, Event) else event

        for _event in _events:
            self.counts[_event] += 1

            if handler := self.handlers.get(_event):
                handler(*args, **kwargs)
                for fwd in self.forwards.get(_event, []):
                    logger.debug(f"Forwarding {_event} to {fwd}")
                    self.emit(fwd, *args, **kwargs)

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
            self.counts[event] += 1

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

    def has(self, name: str, *, event: Event | None = None) -> bool:
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

    def remove(
        self,
        name: str,
        *,
        event: Event[P] | Iterable[Event[P]] | None = None,
    ) -> bool:
        """Remove any callback(s) registered with `name` from the event handlers.

        Args:
            name: The name of the callback(s) to remove.
            event: The event to remove the callback(s) from. If `None` then
                the callback will be removed from all events.

        Returns:
            True if a callback was removed, False otherwise.
        """
        if event is None:
            return self.remove(name, event=self.events)

        _events = [event] if isinstance(event, Event) else event

        any_removed = False
        for _event in _events:
            handler = self.handlers.get(_event)
            if handler is None:
                raise KeyError(f"Event {_event} not found.")

            if name in handler:
                logger.debug(f"{self.name}: Removing {name} handler from {event}")
                del handler[name]
                any_removed = True

        return any_removed

    def forward(self, frm: Event, to: Event) -> None:
        """Forward an event to another event.

        Args:
            frm: The event to forward.
            to: The event to forward to.
        """
        self.forwards[frm].append(to)

    def subscriber(self, *event: Event[P]) -> Subscriber[P]:
        """Create a subscriber for an event.

        Args:
            event: The event(s) that will be subscribed to

        Returns:
            A subscriber for the event.
        """
        _event = event[0] if len(event) == 1 else event
        return Subscriber(self, _event)  # type: ignore


class Emitter:
    """An event emitter.

    This is a convenience class that wraps an event manager and provides
    a way to emit events. The events emitter and subscribed to will be
    identified by a UUID, such that two objects emitting the same event
    will have a different set of listeners who will be called. For
    downstream users, this means they must subscribe to events directly
    from the object they are using.
    """

    def __init__(self, event_manager: EventManager | str | None = None) -> None:
        """Initialise the emitter.

        Args:
            event_manager: The event manager to use. If a `str` is given
                then a new event manager will be created with the given
                name. If the event manager is `None` then a new
                event manager will be created with a random uuid as the name.
        """
        super().__init__()
        if isinstance(event_manager, str):
            event_manager = EventManager(name=event_manager)
        elif event_manager is None:
            event_manager = EventManager(name=f"Emitter-{uuid4()}")

        self.event_manager = event_manager
        self.emitted_events: set[Event] = set()

    def emit(
        self,
        event: Event[P] | Iterable[Event[P]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
            args: The positional arguments to pass to the event handlers.
            kwargs: The keyword arguments to pass to the event handlers.
        """
        if isinstance(event, Event):
            self.event_manager.emit(event, *args, **kwargs)
            return

        self.event_manager.emit_many({e: (args, kwargs) for e in event})

    def emit_many(
        self,
        events: dict[Event, tuple[tuple[Any, ...] | None, dict[str, Any] | None]],
    ) -> None:
        """Emit many events at once.

        This is useful for cases where you don't want to favour one callback
        over another, and so uses the time a callback was registered to call
        the callback instead.

        ```python
        task.emit_many({
            "event1": ((1, 2, 3), {"x": 4, "y": 5}), # (1)!
            "event2": (("hello", "world"), None), # (2)!
            "event3": (None, None),  # (3)!
        })
        ```
        1. Pass the positional and keyword arguments as a tuple and a dictionary
        2. Specify None for the keyword arguments if you don't want to pass any.
        3. Specify None for both if you don't want to pass any arguments to the event

        Args:
            events: A dictionary of events to emit. The keys are the events
                to emit, and the values are tuples of the positional and
                keyword arguments to pass to the event handlers.
        """
        self.event_manager.emit_many(
            dict(events.items()),
        )

    def subscriber(self, *event: Event[P]) -> Subscriber[P]:
        """Create a subscriber for an event.

        Args:
            event: The event(s) that will be subscribed to

        Returns:
            A subscriber for the event.
        """
        return self.event_manager.subscriber(*event)

    @property
    def event_counts(self) -> dict[Event, int]:
        """The event counter.

        Useful for predicates, for example
        ```python
        from amltk.scheduling import Task

        my_scheduler.on_task_finished(
            do_something,
            when=lambda sched: sched.event_counts[Task.FINISHED] > 10
        )
        ```
        """
        return dict(self.event_manager.counts)

    @overload
    def on(
        self,
        event: Event[P] | Iterable[Event[P]],
        callback: None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> SubcriberDecorator[P]:
        ...

    @overload
    def on(
        self,
        event: Event[P] | Iterable[Event[P]],
        callback: Callable[P, Any] | Iterable[Callable[P, Any]],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> None:
        ...

    def on(
        self,
        event: Event[P] | Iterable[Event[P]],
        callback: Callable[P, Any] | Iterable[Callable[P, Any]] | None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        every: int | None = None,
        repeat: int = 1,
        limit: int | None = None,
    ) -> None | SubcriberDecorator[P]:
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
        if callback is None:
            event = (event,) if isinstance(event, Event) else tuple(event)
            subscriber = Subscriber(manager=self.event_manager, event=event)
            return subscriber(
                callback,
                name=name,
                when=when,
                every=every,
                repeat=repeat,
                limit=limit,
            )

        self.event_manager.on(
            event,
            callback,
            name=name,
            when=when,
            every=every,
            repeat=repeat,
            limit=limit,
        )
        return None
