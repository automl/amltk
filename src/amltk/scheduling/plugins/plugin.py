r"""A plugin that can be attached to a Task.

By inheriting from a `Plugin`, you can hook into a
[`Task`][amltk.scheduling.Task]. A plugin can affect, modify and extend its
behaviours. Please see the documentation of the methods for more information.
Creating a plugin is only necesary if you need to modify actual behaviour of
the task. For siply hooking into the lifecycle of a task, you can use the `@events`
that a `Task` emits.

??? example "Creating a Plugin"

    For a full example of a simple plugin, see the
    [`Limiter`][amltk.scheduling.plugins.Limiter] plugin which prevents
    the task being submitted if for example, it has already been submitted
    too many times.

    The below example shows how to create a plugin that prints the task name
    before submitting it. It also emits an event when the task is submitted.

    ```python exec="true" source="material-block" html="true"
    from __future__ import annotations
    from typing import Callable

    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Plugin
    from amltk.scheduling.events import Event

    # A simple plugin that prints the task name before submitting
    class Printer(Plugin):
        name = "my-plugin"

        # Define an event the plugin will emit
        # Event[Task] indicates the callback for the event will be called with the task
        PRINTED: Event[str] = Event("printer-msg")

        def __init__(self, greeting: str):
            self.greeting = greeting
            self.n_greetings = 0

        def attach_task(self, task) -> None:
            self.task = task
            # Register an event with the task, this lets the task know valid events
            # people can subscribe to and helps it show up in visuals
            task.emitter.add_event(self.PRINTED)
            task.on_submitted(self._print_submitted, hidden=True)  # You can hide this callback from visuals

        def pre_submit(self, fn, *args, **kwargs) -> tuple[Callable, tuple, dict]:
            print(f"{self.greeting} for {self.task} {args} {kwargs}")
            self.n_greetings += 1
            return fn, args, kwargs

        def _print_submitted(self, future, *args, **kwargs) -> None:
            msg = f"Task was submitted {self.task} {args} {kwargs}"
            self.task.emitter.emit(self.PRINTED, msg)  # Emit the event with a msg

        def copy(self) -> Printer:
            # Plugins need to be able to copy themselves as if fresh
            return self.__class__(self.greeting)

        def __rich__(self):
            # Custome how the plugin is displayed in rich (Optional)
            # rich is an optional dependancy of amltk so we move the imports into here
            from rich.panel import Panel

            return Panel(
                f"Greeting: {self.greeting} ({self.n_greetings})",
                title=f"Plugin {self.name}"
            )

    def fn(x: int) -> int:
        return x + 1
    from amltk._doc import make_picklable; make_picklable(fn)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(fn, plugins=[Printer("Hello")])

    @scheduler.on_start
    def on_start():
        task.submit(15)

    @task.on("printer-msg")
    def callback(msg: str):
        print("\nmsg")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```

    All methods are optional, and you can choose to implement only the ones
    you need. Most plugins will likely need to implement the
    [`attach_task()`][amltk.scheduling.Plugin.attach_task] method, which is called
    when the plugin is attached to a task. In this method, you can for
    example subscribe to events on the task, create new subscribers for people
    to use or even store a reference to the task for later use.

    Plugins are also encouraged to utilize the events of a
    [`Task`][amltk.scheduling.Task] to further hook into the lifecycle of the task.
    For exampe, by saving a reference to the task in the `attach_task()` method, you
    can use the [`emit()`][amltk.scheduling.Task] method of the task to emit
    your own specialized events.
"""  # noqa: E501
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, ClassVar, TypeVar
from typing_extensions import ParamSpec, Self, override

from amltk._richutil.renderable import RichRenderable
from amltk.scheduling.events import Event

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling import Task

logger = logging.getLogger(__name__)


P = ParamSpec("P")
P2 = ParamSpec("P2")

R = TypeVar("R")
R2 = TypeVar("R2")
CallableT = TypeVar("CallableT", bound=Callable)


class Plugin(RichRenderable, ABC):
    """A plugin that can be attached to a Task."""

    name: ClassVar[str]
    """The name of the plugin.

    This is used to identify the plugin during logging.
    """

    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task.

        This method is called when the plugin is attached to a task. This
        is the place to subscribe to events on the task, create new subscribers
        for people to use or even store a reference to the task for later use.

        Args:
            task: The task the plugin is being attached to.
        """

    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict] | None:
        """Pre-submit hook.

        This method is called before the task is submitted.

        Args:
            fn: The task function.
            *args: The arguments to the task function.
            **kwargs: The keyword arguments to the task function.

        Returns:
            A tuple of the task function, arguments and keyword arguments
            if the task should be submitted, or `None` if the task should
            not be submitted.
        """
        return fn, args, kwargs

    def events(self) -> list[Event]:
        """Return a list of events that this plugin emits.

        Likely no need to override this method, as it will automatically
        return all events defined on the plugin.
        """
        inherited_attrs = chain.from_iterable(
            vars(cls).values() for cls in self.__class__.__mro__
        )
        return [attr for attr in inherited_attrs if isinstance(attr, Event)]

    @abstractmethod
    def copy(self) -> Self:
        """Return a copy of the plugin.

        This method is used to create a copy of the plugin when a task is
        copied. This is useful if the plugin stores a reference to the task
        it is attached to, as the copy will need to store a reference to the
        copy of the task.
        """
        ...

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel

        return Panel("", title=f"Plugin {self.name}")
