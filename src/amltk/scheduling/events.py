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

    === "`on('event', limit=...)`"

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

        @scheduler.on_future_result(limit=3)
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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload
from typing_extensions import ParamSpec, override
from uuid import uuid4

from amltk._functional import callstring, funcname
from amltk._richutil.renderers.function import Function

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

    def __call__(
        self,
        callback: Callable[P, Any] | None = None,
        *,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int = 1,
        hidden: bool = False,
    ) -> Callable[P, Any] | partial[Callable[P, Any]]:
        """Subscribe to the event associated with this object.

        Args:
            callback: The callback to register.
            when: A predicate that must be satisfied for the callback to be called.
            every: The callback will be called every `every` times the event is emitted.
            repeat: The callback will be called `repeat` times successively.
            limit: The maximum number of times the callback can be called.
            hidden: Whether to hide the callback in visual output.
                This is mainly used to facilitate Plugins who
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
        self.n_calls_to_handler += 1
        if self.every > 1 and self.n_calls_to_handler % self.every != 0:
            return

        if self.when is not None and not self.when():
            return

        limit = self.limit if self.limit is not None else math.inf
        for _ in range(self.repeat):
            if self.n_calls_to_callback >= limit:
                return

            logger.debug(f"Calling: {callstring(self.callback)}")
            self.callback(*args, **kwargs)
            self.n_calls_to_callback += 1

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


class Emitter(Mapping[Event, list[Handler]]):
    """An event emitter.

    This class is used to emit events and register callbacks for those events.
    It also provides a convenience function
    [`subscriber()`][amltk.scheduling.events.Emitter.subscriber] such
    that objects using an `Emitter` can easily create access points for users
    to directly subscribe to their [`Events`][amltk.scheduling.events.Event].
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
        callback: Callable,
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
                This is mainly used to facilitate Plugins who
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
        self.handlers[event].append(
            Handler(
                callback,
                when=when,
                every=every,
                repeat=repeat,
                limit=limit,
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

    def __rich__(self) -> Tree:
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
