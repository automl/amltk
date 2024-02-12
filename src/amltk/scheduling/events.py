"""One of the primary ways to respond to `@events` emitted
with by a [`Task`][amltk.scheduling.Task]
the [`Scheduler`][amltk.scheduling.Scheduler]
is through use of a **callback**.

The reason for this is to enable an easier time for API's to utilize
multiprocessing and remote compute from the `Scheduler`, without having
to burden users with knowing the details of how to use multiprocessing.

A callback subscribes to some event using a decorator but can also be done in
a functional style if preferred. The below example is based on the
event [`@scheduler.on_start`][amltk.scheduling.Scheduler.on_start] but
the same applies to all events.

=== "Decorators"

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    @scheduler.on_start
    def print_hello() -> None:
        print("hello")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, fontsize="small")  # markdown-exec: hide
    ```

=== "Functional"

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    def print_hello() -> None:
        print("hello")

    scheduler.on_start(print_hello)
    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, fontsize="small")  # markdown-exec: hide
    ```

There are a number of ways to customize the behaviour of these callbacks, notably
to control how often they get called and when they get called.

??? tip "Callback customization"


    === "`on('event', repeat=...)`"

        This will cause the callback to be called `repeat` times successively.
        This is most useful in combination with
        [`@scheduler.on_start`][amltk.scheduling.Scheduler.on_start] to launch
        a number of tasks at the start of the scheduler.

        ```python exec="true" source="material-block" html="true" hl_lines="11"
        from amltk import Scheduler

        N_WORKERS = 2

        def f(x: int) -> int:
            return x * 2
        from amltk._doc import make_picklable; make_picklable(f)  # markdown-exec: hide

        scheduler = Scheduler.with_processes(N_WORKERS)
        task = scheduler.task(f)

        @scheduler.on_start(repeat=N_WORKERS)
        def on_start():
            task.submit(1)

        scheduler.run()
        from amltk._doc import doc_print; doc_print(print, scheduler, fontsize="small")  # markdown-exec: hide
        ```

    === "`on('event', max_calls=...)`"

        Limit the number of times a callback can be called, after which, the callback
        will be ignored.

        ```python exec="true" source="material-block" html="True" hl_lines="13"
        from asyncio import Future
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(2)

        def expensive_function(x: int) -> int:
            return x ** 2
        from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

        @scheduler.on_start
        def submit_calculations() -> None:
            scheduler.submit(expensive_function, 2)

        @scheduler.on_future_result(max_calls=3)
        def print_result(future, result) -> None:
            scheduler.submit(expensive_function, 2)

        scheduler.run()
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

    === "`on('event', when=...)`"

        A callable which takes no arguments and returns a `bool`. The callback
        will only be called when the `when` callable returns `True`.

        Below is a rather contrived example, but it shows how we can use the
        `when` parameter to control when the callback is called.

        ```python exec="true" source="material-block" html="True" hl_lines="8 12"
        import random
        from amltk.scheduling import Scheduler

        LOCALE = random.choice(["English", "German"])

        scheduler = Scheduler.with_processes(1)

        @scheduler.on_start(when=lambda: LOCALE == "English")
        def print_hello() -> None:
            print("hello")

        @scheduler.on_start(when=lambda: LOCALE == "German")
        def print_guten_tag() -> None:
            print("guten tag")

        scheduler.run()
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

    === "`on('event', every=...)`"

        Only call the callback every `every` times the event is emitted. This
        includes the first time it's called.

        ```python exec="true" source="material-block" html="True" hl_lines="6"
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(1)

        # Print "hello" only every 2 times the scheduler starts.
        @scheduler.on_start(every=2)
        def print_hello() -> None:
            print("hello")

        # Run the scheduler 5 times
        scheduler.run()
        scheduler.run()
        scheduler.run()
        scheduler.run()
        scheduler.run()
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

### Emitter, Subscribers and Events
This part of the documentation is not necessary to understand or use for AMLTK. People
wishing to build tools upon AMLTK may still find this a useful component to add to their
arsenal.

The core of making this functionality work is the [`Emitter`][amltk.scheduling.events.Emitter].
Its purpose is to have `@events` that can be emitted and subscribed to. Classes like the
[`Scheduler`][amltk.scheduling.Scheduler] and [`Task`][amltk.scheduling.Task] carry
around with them an `Emitter` to enable all of this functionality.

Creating an `Emitter` is rather straight-forward, but we must also create
[`Events`][amltk.scheduling.events.Event] that people can subscribe to.

```python
from amltk.scheduling import Emitter, Event
emitter = Emitter("my-emitter")

event: Event[int] = Event("my-event") # (1)!

@emitter.on(event)
def my_callback(x: int) -> None:
    print(f"Got {x}!")

emitter.emit(event, 42) # (2)!
```

1. The typing `#!python Event[int]` is used to indicate that the event will be emitting
    an integer. This is not necessary, but it is useful for type-checking and
    documentation.
2. The `#!python emitter.emit(event, 42)` is used to emit the event. This will call
    all the callbacks registered for the event, i.e. `#!python my_callback()`.

!!! warning "Independent Events"

    Given a single `Emitter` and a single instance of an `Event`, there is no way to
    have different `@events` for callbacks. There are two options, both used extensively
    in AMLTK.

    The first is to have different `Events` quite naturally, i.e. you distinguish
    between different things that can happen. However, you often want to have different
    objects emit the same `Event` but have different callbacks for each object.

    This makes most sense in the context of a `Task` the `Event` instances are shared as
    class variables in the `Task` class, however a user likely want's to subscribe to
    the `Event` for a specific instance of the `Task`.

    This is where the second option comes in, in which each object carries around its
    own `Emitter` instance. This is how a user can subscribe to the same kind of `Event`
    but individually for each `Task`.


However, to shield users from this and to create named access points for users to
subscribe to, we can use the [`Subscriber`][amltk.scheduling.events.Subscriber] class,
conveniently created by the [`Emitter.subscriber()`][amltk.scheduling.events.Emitter.subscriber]
method.

```python
from amltk.scheduling import Emitter, Event
emitter = Emitter("my-emitter")

class GPT:

    event: Event[str] = Event("my-event")

    def __init__(self) -> None:
        self.on_answer: Subscriber[str] = emitter.subscriber(self.event)

    def ask(self, question: str) -> None:
        emitter.emit(self.event, "hello world!")

gpt = GPT()

@gpt.on_answer
def print_answer(answer: str) -> None:
    print(answer)

gpt.ask("What is the conical way for an AI to greet someone?")
```

Typically these event based systems make little sense in a synchronous context, however
with the [`Scheduler`][amltk.scheduling.Scheduler] and [`Task`][amltk.scheduling.Task]
classes, they are used to enable a simple way to use multiprocessing and remote compute.
"""  # noqa: E501
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, overload
from typing_extensions import ParamSpec, override

from more_itertools import first_true

from amltk._functional import funcname
from amltk._richutil.renderers.function import Function
from amltk.exceptions import EventNotKnownError
from amltk.randomness import randuid

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.text import Text

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
                Iterable[Handler[P, R]],
                tuple[Any, ...] | None,
                dict[str, Any] | None,
            ]
        ],
    ) -> list[tuple[Handler[P, R], R | None]]:
        """Call all events in the scheduler."""
        all_handlers = []
        for handlers, args, kwargs in events:
            all_handlers += [
                (handler, args or (), kwargs or {}) for handler in handlers
            ]

        sorted_handlers = sorted(all_handlers, key=lambda item: item[0].registered_at)
        return [
            (handler, handler(*args, **kwargs))
            for handler, args, kwargs in sorted_handlers
        ]


@dataclass(frozen=True)
class Event(Generic[P, R]):
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
class Subscriber(Generic[P, R]):
    """An object that can be used to easily subscribe to a certain event.

    ```python
    from amltk.scheduling.events import Event, Subscriber

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
    event: Event[P, R]

    @property
    def event_counts(self) -> int:
        """The number of times this event has been emitted."""
        return self.emitter.event_counts[self.event]

    def register(
        self,
        callback: Callable[P, R],
        *,
        when: Callable[[], bool] | None = None,
        max_calls: int | None = None,
        repeat: int = 1,
        every: int = 1,
        hidden: bool = False,
    ) -> None:
        """Register a callback for this subscriber.

        Args:
            callback: The callback to register.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            max_calls: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate Plugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.
        """
        self.emitter.register(
            event=self.event,
            callback=callback,
            when=when,
            max_calls=max_calls,
            repeat=repeat,
            every=every,
            hidden=hidden,
        )

    @overload
    def __call__(
        self,
        callback: None = None,
        *,
        when: Callable[[], bool] | None = ...,
        max_calls: int | None = ...,
        repeat: int = ...,
        every: int = ...,
        hidden: bool = ...,
    ) -> Callable[[Callable[P, R]], None]:
        ...

    @overload
    def __call__(
        self,
        callback: Callable[P, R],
        *,
        when: Callable[[], bool] | None = ...,
        max_calls: int | None = ...,
        repeat: int = ...,
        every: int = ...,
        hidden: bool = ...,
    ) -> None:
        ...

    def __call__(
        self,
        callback: Callable[P, R] | None = None,
        *,
        when: Callable[[], bool] | None = None,
        max_calls: int | None = None,
        repeat: int = 1,
        every: int = 1,
        hidden: bool = False,
    ) -> Callable[[Callable[P, R]], None] | None:
        """A decorator to register a callback for this subscriber.

        Args:
            callback: The callback to register. If `None`, then this
                acts as a decorator, as you would normally use it. Prefer
                to leave this as `None` and use
                [`register()`][amltk.scheduling.events.Subscriber.register] if
                you have a direct reference to the function and are not decorating it.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            max_calls: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate Plugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.
        """
        if callback is None:
            return partial(
                self.register,
                when=when,
                max_calls=max_calls,
                repeat=repeat,
                every=every,
                hidden=hidden,
            )
        self.register(
            callback,
            when=when,
            max_calls=max_calls,
            repeat=repeat,
            every=every,
            hidden=hidden,
        )
        return None

    def emit(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[tuple[Handler[P, R], R | None]]:
        """Emit this subscribers event."""
        return self.emitter.emit(self.event, *args, **kwargs)


@dataclass
class Handler(Generic[P, R]):
    """A handler for an event.

    This is a simple class that holds a callback and any predicate
    that must be satisfied for it to be triggered.
    """

    callback: Callable[P, R]
    when: Callable[[], bool] | None = None
    every: int = 1
    n_calls_to_handler: int = 0
    n_calls_to_callback: int = 0
    max_calls: int | None = None
    repeat: int = 1
    registered_at: int = field(default_factory=time.time_ns)
    hidden: bool = False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        """Call the callback if the predicate is satisfied.

        If the predicate is not satisfied, then `None` is returned.
        """
        self.n_calls_to_handler += 1
        if self.every > 1 and self.n_calls_to_handler % self.every != 0:
            return None

        if self.when is not None and not self.when():
            return None

        max_calls = self.max_calls if self.max_calls is not None else math.inf

        if self.repeat == 1:
            if self.n_calls_to_callback >= max_calls:
                return None

            self.n_calls_to_callback += 1
            return self.callback(*args, **kwargs)

        if self.n_calls_to_callback >= max_calls:
            return None

        responses = iter(self.callback(*args, **kwargs) for _ in range(self.repeat))
        self.n_calls_to_callback += 1
        first_response = next(responses)
        if first_response is not None:
            raise ValueError("A callback with a response cannot have `repeat` > 1.")

        # Otherwise just exhaust the iterator
        list(responses)
        return None

    def __rich__(self) -> Text:
        from rich.text import Text

        f_rich = Function(self.callback).__rich__()
        if self.n_calls_to_callback == 0:
            return f_rich

        return Text.assemble(
            f_rich,
            Text(" (", style="italic"),
            Text(f"{self.n_calls_to_callback}", style="yellow italic"),
            Text(")", style="italic"),
        )


class Emitter:
    """An event emitter.

    This class is used to emit events and register callbacks for those events.
    It also provides a convenience function
    [`subscriber()`][amltk.scheduling.events.Emitter.subscriber] such
    that objects using an `Emitter` can easily create access points for users
    to directly subscribe to their [`Events`][amltk.scheduling.events.Event].
    """

    HandlerResponses: TypeAlias = Iterable[tuple[Handler[P, R], R | None]]
    """The stream of responses from handlers when an event is triggered."""

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
        self.unique_ref = f"{name}-{randuid()}"
        self.emitted_events: set[Event] = set()

        self.name = name
        self.handlers = defaultdict(list)
        self.event_counts = Counter()

    def emit(
        self,
        event: Event[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[tuple[Handler[P, R], R | None]]:
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
        return [(handler, handler(*args, **kwargs)) for handler in self.handlers[event]]

    @property
    def events(self) -> list[Event]:
        """Return a list of the events."""
        return list(self.handlers.keys())

    def subscriber(self, event: Event[P, R]) -> Subscriber[P, R]:
        """Create a subscriber for an event.

        Args:
            event: The event to register the callback for.
        """
        if event not in self.handlers:
            self.handlers[event] = []

        return Subscriber(self, event)

    @overload
    def on(
        self,
        event: str,
        *,
        when: Callable[[], bool] | None = ...,
        every: int = ...,
        repeat: int = ...,
        max_calls: int | None = ...,
        hidden: bool = ...,
    ) -> Callable[[Callable[..., Any | None]], None]:
        ...

    @overload
    def on(
        self,
        event: Event[P, R],
        *,
        when: Callable[[], bool] | None = ...,
        every: int = ...,
        repeat: int = ...,
        max_calls: int | None = ...,
        hidden: bool = ...,
    ) -> Callable[[Callable[P, R | None]], None]:
        ...

    def on(
        self,
        event: Event[P, R] | str,
        *,
        when: Callable[[], bool] | None = None,
        every: int = 1,
        repeat: int = 1,
        max_calls: int | None = None,
        hidden: bool = False,
    ) -> Callable[[Callable[P, R | None]], None]:
        """Register a callback for an event as a decorator.

        Args:
            event: The event to register the callback for.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            max_calls: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate Plugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.
        """
        return partial(
            self.subscriber(event=self.as_event(event)),  # type: ignore
            when=when,
            every=every,
            repeat=repeat,
            max_calls=max_calls,
            hidden=hidden,
        )

    def register(
        self,
        event: Event[P, R] | str,
        callback: Callable[P, R],
        *,
        when: Callable[[], bool] | None = None,
        every: int = 1,
        repeat: int = 1,
        max_calls: int | None = None,
        hidden: bool = False,
    ) -> None:
        """Register a callback for an event.

        Args:
            event: The event to register the callback for.
            callback: The callback to register.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            max_calls: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate Plugins who
                act upon events but don't want to be seen, primarily
                as they are just book-keeping callbacks.
        """
        event = self.as_event(event)

        if repeat <= 0:
            raise ValueError(f"{repeat=} must be a positive integer.")

        if every <= 0:
            raise ValueError(f"{every=} must be a positive integer.")

        # Make sure it shows up in the event counts, setting it to 0 if it
        # doesn't exist
        self.event_counts.setdefault(event, 0)
        self.handlers[event].append(
            Handler(
                callback,
                when=when,
                every=every,
                repeat=repeat,
                max_calls=max_calls,
                hidden=hidden,
            ),
        )

        _name = funcname(callback)
        msg = f"{self.name}: Registered callback '{_name}' for event {event}"
        if every > 1:
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

    def __rich__(self) -> RenderableType:
        from rich.tree import Tree

        tree = Tree(self.name or "", hide_root=self.name is None)

        # This just groups events with callbacks together
        handler_items = sorted(
            self.handlers.items(),
            key=lambda item: not any(item[1])
            or not all(handler.hidden for handler in item[1]),
        )

        for event, _handlers in handler_items:
            event_text = event.__rich__()
            if (count := self.event_counts[event]) >= 1:
                event_text.append(f" {count}", style="yellow italic")

            event_tree = tree.add(event_text)
            for handler in _handlers:
                if not handler.hidden:
                    event_tree.add(handler)

        return tree

    @overload
    def as_event(self, key: str) -> Event:
        ...

    @overload
    def as_event(self, key: Event[P, R]) -> Event[P, R]:
        ...

    def as_event(self, key: str | Event) -> Event:
        """Return the event associated with the key."""
        match key:
            case Event():
                return key
            case str():
                match = first_true(self.events, None, lambda e: e.name == key)
                if match is None:
                    raise EventNotKnownError(
                        f"{key=} is not a valid event for {self.name}."
                        f"\nKnown events are: {[e.name for e in self.events]}",
                    )
                return match
            case _:
                raise TypeError(f"{key=} must be a string or an Event.")
