"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.

There is also the [`CommTask`][byop.scheduling.comm_task.CommTask] which can
be used for communication between the task and the main process.
"""
from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Generic, Hashable, Iterable, TypeVar
from typing_extensions import ParamSpec

from byop.events import Event, Subscriber
from byop.functional import callstring, funcname

if TYPE_CHECKING:
    from concurrent.futures import Future

    from byop.scheduling.scheduler import Scheduler
    from byop.scheduling.task_plugin import TaskPlugin

logger = logging.getLogger(__name__)


P = ParamSpec("P")
P2 = ParamSpec("P2")

R = TypeVar("R")
CallableT = TypeVar("CallableT", bound=Callable)


class Task(Generic[P, R]):
    """A task is a unit of work that can be scheduled by the scheduler.

    It is defined by its `name` and a `function` to call. Whenever a task
    has its `__call__` method called, the function will be dispatched to run
    by a [`Scheduler`][byop.scheduling.scheduler.Scheduler].

    The scheduler will emit specific events
    to this task which look like `(task.name, TaskEvent)`.

    To interact with the results of these tasks, you must subscribe to to these
    events and provide callbacks.

    ```python hl_lines="9"
    # Define some function to run
    def f(x: int) -> int:
        return x * 2

    # And a scheduler to run it on
    scheduler = Scheduler.with_processes(2)

    # Create the task object, the type anotation Task[[int], int] isn't required
    my_task: Task[[int], int] = scheduler.task("call_f", f)

    # Subscribe to events
    my_task.on_returned(lambda result: print(result))
    my_task.on_exception(lambda error: print(error))
    ```

    Attributes:
        name: The name of the task.
        function: The function of this task
        scheduler: The scheduler that this task is registered with.
    """

    SUBMITTED: Event[Future] = Event("task-submitted")
    """A Task has been submitted to the scheduler."""

    DONE: Event[Future] = Event("task-done")
    """A Task has finished running."""

    CANCELLED: Event[Future] = Event("task-cancelled")
    """A Task has been cancelled."""

    F_RETURNED: Event[Future, Any] = Event("task-future-returned")
    """A Task has successfully returned a value. Comes with Future"""

    RETURNED: Event[[Any]] = Event("task-returned")
    """A Task has successfully returned a value."""

    F_EXCEPTION: Event[Future, BaseException] = Event("task-future-exception")
    """A Task failed to return anything but an exception. Comes with Future"""

    EXCEPTION: Event[BaseException] = Event("task-exception")
    """A Task failed to return anything but an exception."""

    def __init__(
        self,
        function: Callable[P, R],
        scheduler: Scheduler,
        *,
        name: str | None = None,
        plugins: Iterable[TaskPlugin] = (),
    ) -> None:
        """Initialize a task.

        Args:
            name: The name of the task.
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            plugins: The plugins to use for this task.
        """
        self.plugins = plugins
        self.name = funcname(function) if name is None else name

        # We hold reference to this because we might possible wrap it
        self._original_function = function

        self.scheduler = scheduler

        self.queue: list[Future[R]] = []

        # Set up subscription methods to events
        self.on_submitted: Subscriber[Future[R]]
        self.on_submitted = self.subscriber(self.SUBMITTED)

        self.on_done: Subscriber[Future[R]]
        self.on_done = self.subscriber(self.DONE)

        self.on_cancelled: Subscriber[Future[R]]
        self.on_cancelled = self.subscriber(self.CANCELLED)

        self.on_f_returned: Subscriber[Future[R], R]
        self.on_f_returned = self.subscriber(self.F_RETURNED)

        self.on_f_exception: Subscriber[Future[R], BaseException]
        self.on_f_exception = self.subscriber(self.F_EXCEPTION)

        self.on_returned: Subscriber[R]
        self.on_returned = self.subscriber(self.RETURNED)

        self.on_exception: Subscriber[BaseException]
        self.on_exception = self.subscriber(self.EXCEPTION)

        for plugin in self.plugins:
            plugin.attach(self)
            function = plugin.wrap(function)  # type: ignore

        self.function = function

    def futures(self) -> list[Future[R]]:
        """Get the futures for this task.

        Returns:
            A list of futures for this task.
        """
        return self.queue

    @property
    def counts(self) -> dict[Event, int]:
        """Get the number of event counts for this task.

        Returns:
            A counter of the number of times each event has been emitted.
        """
        return {
            event: count
            for event in self.events()
            if (count := self.scheduler.event_manager.counts.get((event, self.name)))
            is not None
            and count > 0
        }

    def events(self) -> list[Event]:
        """Return the events that this task could emit."""
        inherited_attrs = chain.from_iterable(
            vars(cls).values() for cls in self.__class__.__mro__
        )
        inherited_events = [attr for attr in inherited_attrs if isinstance(attr, Event)]
        plugin_events = chain.from_iterable(plugin.events() for plugin in self.plugins)
        return inherited_events + list(plugin_events)

    def subscriber(self, event: Event[P2]) -> Subscriber[P2]:
        """Return an object that can be used to subscribe to an event for this task.

        Args:
            event: The event to subscribe to.

        Returns:
            A callable object that can be used to subscribe to events.
        """
        _event = (event, self.name)
        return self.scheduler.event_manager.subscriber(_event)

    def emit(self, event: Event[P2], *args: P2.args, **kwargs: P2.kwargs) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
            *args: The positional arguments to pass to the event handlers.
            **kwargs: The keyword arguments to pass to the event handlers.
        """
        task_event = (event, self.name)
        self.scheduler.event_manager.emit(task_event, *args, **kwargs)

    def emit_many(
        self,
        events: dict[Hashable, tuple[tuple[Any, ...] | None, dict[str, Any] | None]],
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
        ```
        1. Pass the positional and keyword arguments as a tuple and a dictionary
        2. Specify None for the keyword arguments if you don't want to pass any.
        3. Specify None for both if you don't want to pass any arguments to the event

        Args:
            events: A dictionary of events to emit. The keys are the events
                to emit, and the values are tuples of the positional and
                keyword arguments to pass to the event handlers.
        """
        self.scheduler.event_manager.emit_many(
            {(event, self.name): params for event, params in events.items()},
        )

    def forward_event(self, frm: Hashable, to: Hashable) -> None:
        """Forward an event to another event.

        Args:
            frm: The event to forward.
            to: The event to forward to.
        """
        self.scheduler.event_manager.forward(frm, to)

    @property
    def n_running(self) -> int:
        """Get the number of futures for this task that are currently running."""
        return sum(1 for f in self.queue if not f.done())

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Future[R] | None:
        """Dispatch this task.

        Args:
            *args: The positional arguments to pass to the task.
            **kwargs: The keyword arguments to call the task with.

        Returns:
            The future of the task, or `None` if the limit was reached.
        """
        # Inform all plugins that the task is about to be called
        # They have chance to cancel submission based on their return
        # value.
        for plugin in self.plugins:
            should_submit = plugin.pre_submit(self.function, *args, **kwargs)

            if not should_submit:
                logger.debug(
                    f"Plugin '{plugin.name}' prevented {self} from being submitted"
                    f" with {callstring(self._original_function, *args, **kwargs)}",
                )
                return None

        future = self.scheduler.submit(self.function, *args, **kwargs)
        if future is None:
            msg = (
                f"Task {callstring(self._original_function, *args, **kwargs)} was not"
                " able to be submitted. The scheduler is likely already finished."
            )
            logger.info(msg)
            return None

        self.queue.append(future)

        # We actuall have the function wrapped in something will
        # attach tracebacks to errors, so we need to get the
        # original function name.
        msg = (
            f"Submitted {self} with "
            f"{callstring(self._original_function, *args, **kwargs)}"
        )
        logger.debug(msg)
        self.emit(Task.SUBMITTED, future)

        # Process the task once it's completed
        future.add_done_callback(self._process_future)

        return future

    def _process_future(self, future: Future[R]) -> None:
        try:
            self.queue.remove(future)
        except ValueError as e:
            raise ValueError(f"{future=} not found in task queue {self.queue=}") from e

        if future.cancelled():
            self.emit(Task.CANCELLED, future)
            return

        self.emit(Task.DONE, future)

        exception = future.exception()
        if exception is not None:
            self.emit_many(
                {
                    Task.F_EXCEPTION: ((future, exception), None),
                    Task.EXCEPTION: ((exception,), None),
                },
            )
        else:
            result = future.result()
            self.emit_many(
                {
                    Task.F_RETURNED: ((future, result), None),
                    Task.RETURNED: ((result,), None),
                },
            )

    def __repr__(self) -> str:
        kwargs = {
            k: v
            for k, v in [("name", self.name), ("plugins", self.plugins)]
            if v is not None
        }
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
