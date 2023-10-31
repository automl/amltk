"""All code for allowing an event system."""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
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

from amltk.functional import callstring, funcname
from amltk.richutil.renderers.function import Function

if TYPE_CHECKING:
    from rich.text import Text
    from rich.tree import Tree

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
            tuple[
                Iterable[Handler[P]],
                tuple[Any, ...] | None,
                dict[str, Any] | None,
            ]
        ],
    ) -> None:
        """Call all events in the scheduler."""
        all_handlers = []
        for handlers, args, kwargs in events:
            all_handlers += [
                (handler, args or (), kwargs or {}) for handler in handlers
            ]

        sorted_handlers = sorted(all_handlers, key=lambda item: item[0].registered_at)
        for handler, args, kwargs in sorted_handlers:
            handler(*args, **kwargs)


@dataclass(frozen=True)
class Event(Generic[P]):
    """An event that can be emitted.

    Attributes:
        name: The name of the event.
    """

    name: str

    @override
    def __eq__(self, other: Any) -> bool:
        """Check if two events are equal."""
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Event):
            return self.name == other.name

        return False

    def __rich__(self) -> Text:
        from rich.text import Text

        return Text("@", style="magenta bold").append(
            self.name,
            style="magenta italic",
        )


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
    when: Callable[[], bool] | None = None
    limit: int | None = None
    repeat: int = 1
    every: int = 1

    @property
    def event_counts(self) -> int:
        """The number of times this event has been emitted."""
        return self.emitter.event_counts[self.event]

    @overload
    def __call__(
        self,
        callback: None = None,
        *,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int = ...,
    ) -> partial[Callable[P, Any]]:
        ...

    @overload
    def __call__(
        self,
        callback: Callable[P, Any],
        *,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int = ...,
        hidden: bool = ...,
    ) -> Callable[P, Any]:
        ...

    @overload
    def __call__(
        self,
        callback: Iterable[Callable[P, Any]],
        *,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int = ...,
        hidden: bool = ...,
    ) -> None:
        ...

    def __call__(
        self,
        callback: Callable[P, Any] | Iterable[Callable[P, Any]] | None = None,
        *,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int = 1,
        hidden: bool = False,
    ) -> Callable[P, Any] | partial[Callable[P, Any]] | None:
        """Subscribe to the event associated with this object.

        Args:
            callback: The callback to register.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            limit: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate TaskPlugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.

        Returns:
            The callback if it was provided, otherwise it acts
                as a decorator.
        """
        if callback is None:
            return partial(
                self.__call__,
                when=when,
                limit=limit,
                repeat=repeat,
                every=every,
            )  # type: ignore

        self.emitter.on(
            self.event,
            callback,
            when=when,
            limit=limit,
            repeat=repeat,
            every=every,
            hidden=hidden,
        )
        if isinstance(callback, Iterable):
            return None
        return callback

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
    every: int = 1
    n_calls_to_handler: int = 0
    n_calls_to_callback: int = 0
    limit: int | None = None
    repeat: int = 1
    registered_at: int = field(default_factory=time.time_ns)
    hidden: bool = False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Call the callback if the predicate is satisfied.

        If the predicate is not satisfied, then `None` is returned.
        """
        limit = self.limit if self.limit is not None else math.inf
        self.n_calls_to_handler += 1
        if self.n_calls_to_handler > limit:
            return

        if self.every > 1 and self.n_calls_to_handler % self.every != 0:
            return

        if self.when is not None and not self.when():
            return

        logger.debug(f"Calling: {callstring(self.callback)}")
        for _ in range(self.repeat):
            self.callback(*args, **kwargs)
            self.n_calls_to_callback += 1

    def __rich__(self) -> Text:
        from rich.text import Text

        f_rich = Function(self.callback, signature=False).__rich__()
        if self.n_calls_to_callback == 0:
            return f_rich

        return Text.assemble(
            f_rich,
            Text(" (", style="italic"),
            Text(f"{self.n_calls_to_callback}", style="yellow italic"),
            Text(")", style="italic"),
        )


class Emitter(Mapping[Event, List[Handler]]):
    """An event emitter.

    This is a convenience class that wraps an event manager and provides
    a way to emit events. The events emitter and subscribed to will be
    identified by a UUID, such that two objects emitting the same event
    will have a different set of listeners who will be called. For
    downstream users, this means they must subscribe to events directly
    from the object they are using.
    """

    name: str | None
    """The name of the emitter."""

    handlers: dict[Event, list[Handler]]
    """A mapping of events to their handlers."""

    event_counts: Counter[Event]
    """A count of all events emitted by this emitter."""

    def __init__(self, name: str | None = None) -> None:
        """Initialise the emitter.

        Args:
            name: The name of the emitter. If not provided, then a UUID
                will be used.
        """
        super().__init__()
        self.unique_ref = f"{name}-{uuid4()}"
        self.emitted_events: set[Event] = set()

        self.name = name
        self.handlers = defaultdict(list)
        self.event_counts = Counter()

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
        for handler in self.handlers[event]:
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
            (handlers, args, kwargs)
            for event, (args, kwargs) in events.items()
            if (handlers := self.get(event)) is not None
        ]

        header = f"{self.name}: Emitting many events"
        logger.debug(header)
        logger.debug(",".join(str(event) for event in events))
        RegisteredTimeCallOrderStrategy.execute(items)

    @override
    def __getitem__(self, event: Event) -> list[Handler]:
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
        when: Callable[[], bool] | None = None,
        every: int = 1,
        repeat: int = 1,
        limit: int | None = None,
    ) -> Subscriber[P]:
        """Create a subscriber for an event.

        Args:
            event: The event to register the callback for.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            limit: The maximum number of times the callback can be called.
        """
        if event not in self.handlers:
            self.handlers[event] = []

        return Subscriber(
            self,
            event,
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
        when: Callable[[], bool] | None = None,
        every: int = 1,
        repeat: int = 1,
        limit: int | None = None,
        hidden: bool = False,
    ) -> None:
        """Register a callback for an event.

        Args:
            event: The event to register the callback for.
            callback: The callback to register.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            limit: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate TaskPlugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.
        """
        if repeat <= 0:
            raise ValueError(f"{repeat=} must be a positive integer.")

        if every <= 0:
            raise ValueError(f"{every=} must be a positive integer.")

        # Make sure it shows up in the event counts, setting it to 0 if it
        # doesn't exist
        self.event_counts.setdefault(event, 0)

        # This hackery is just to get down to a flat list of events that need
        # to be set up
        callbacks = [callback] if callable(callback) else list(callback)
        for c in callbacks:
            self.handlers[event].append(
                Handler(
                    c,
                    when=when,
                    every=every,
                    repeat=repeat,
                    limit=limit,
                    hidden=hidden,
                ),
            )

            _name = funcname(c)
            msg = f"{self.name}: Registered callback '{_name}' for event {event}"
            if every:
                msg += f" every {every} times"
            if when:
                msg += f" with predicate ({funcname(when)})"
            if repeat > 1:
                msg += f" called {repeat} times successively"
            if hidden:
                msg += " (hidden from visual output)"
            logger.debug(msg)

    def add_event(self, *event: Event) -> None:
        """Add an event to the event manager so that it shows up in visuals.

        Args:
            event: The event to add.
        """
        for e in event:
            if e not in self.handlers:
                self.handlers[e] = []

    def __rich__(self) -> Tree:
        from rich.tree import Tree

        tree = Tree(self.name or "", hide_root=self.name is None)

        # This just groups events with callbacks together
        handler_items = sorted(self.handlers.items(), key=lambda item: not any(item[1]))

        for event, _handlers in handler_items:
            event_text = event.__rich__()
            if (count := self.event_counts[event]) >= 1:
                event_text.append(f" {count}", style="yellow italic")

            event_tree = tree.add(event_text)
            for handler in _handlers:
                if not handler.hidden:
                    event_tree.add(handler)

        return tree
