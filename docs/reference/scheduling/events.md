## Events
One of the primary ways to respond to `@events` emitted
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
