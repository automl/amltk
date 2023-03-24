"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.

There is also the [`CommTask`][byop.scheduling.comm_task.CommTask] which can
be used for communication between the task and the main process.
"""

from __future__ import annotations

from asyncio import Future
from itertools import chain
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    ParamSpec,
    TypeVar,
)

from pynisher import Pynisher

from byop.events import Event, Subscriber
from byop.exceptions import exception_wrap
from byop.functional import callstring, funcname

if TYPE_CHECKING:
    from byop.scheduling.scheduler import Scheduler

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

    The scheduler will emit specific [events][byop.scheduling.events.TaskEvent]
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
    my_task.on_return(lambda result: print(result)) # (1)!
    my_task.on_noreturn(lambda error: print(error)) # (2)!
    ```

    1. You could also do: `#!python my_task.on(task.RETURNED, lambda res: print(res))`
    2. You could also do: `#!python my_task.on(task.EXCEPTION, lambda err: print(err))`

    Attributes:
        name: The name of the task.
        function: The function of this task
        scheduler: The scheduler that this task is registered with.
        n_called: How many times this task has been called.
        call_limit: How many times this task can be run. Defaults to `None`task
    """

    SUBMITTED: Event[Future] = Event("task-submitted")
    """A Task has been submitted to the scheduler."""

    DONE: Event[Future] = Event("task-done")
    """A Task has finished running."""

    CANCELLED: Event[Future] = Event("task-cancelled")
    """A Task has been cancelled."""

    RETURNED: Event[Future, Any] = Event("task-returned")
    """A Task has successfully returned a value."""

    EXCEPTION: Event[Future, BaseException] = Event("task-exception")
    """A Task failed to return anything but an exception."""

    TIMEOUT: Event[Future, BaseException] = Event("task-timeout")
    """A Task timed out."""

    MEMORY_LIMIT_REACHED: Event[Future, BaseException] = Event("task-memory-limit")
    """A Task was submitted but reached it's memory limit."""

    CPU_TIME_LIMIT_REACHED: Event[Future, BaseException] = Event("task-cputime-limit")
    """A Task was submitted but reached it's cpu time limit."""

    WALL_TIME_LIMIT_REACHED: Event[Future, BaseException] = Event("task-walltime-limit")
    """A Task was submitted but reached it's wall time limit."""

    CONCURRENT_LIMIT_REACHED: Event[P] = Event("task-concurrent-limit")
    """A Task was submitted but reached it's concurrency limit."""

    CALL_LIMIT_REACHED: Event[P] = Event("task-concurrent-limit")
    """A Task was submitted but reached it's concurrency limit."""

    TimeoutException = Pynisher.TimeoutException
    """The exception that is raised when a task times out."""

    MemoryLimitException = Pynisher.MemoryLimitException
    """The exception that is raised when a task reaches it's memory limit."""

    CpuTimeoutException = Pynisher.CpuTimeoutException
    """The exception that is raised when a task reaches it's cpu time limit."""

    WallTimeoutException = Pynisher.WallTimeoutException
    """The exception that is raised when a task reaches it's wall time limit."""

    def __init__(
        self,
        function: Callable[P, R],
        scheduler: Scheduler,
        *,
        name: str | None = None,
        call_limit: int | None = None,
        concurrent_limit: int | None = None,
        memory_limit: int | tuple[int, str] | None = None,
        cpu_time_limit: int | tuple[float, str] | None = None,
        wall_time_limit: int | tuple[float, str] | None = None,
    ) -> None:
        """Initialize a task.

        Args:
            name: The name of the task.
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            call_limit: How many times this task can be run. Defaults to `None`
            concurrent_limit: How many of this task can be running conccurently.
                By default this is `None` which means that there is no limit.
            memory_limit: The memory limit for this task. Defaults to `None`
            cpu_time_limit: The cpu time limit for this task. Defaults to `None`
            wall_time_limit: The wall time limit for this task. Defaults to `None`
        """
        self.function: Callable[P, R]

        self.name = funcname(function) if name is None else name
        self.scheduler = scheduler
        self.call_limit = call_limit
        self.concurrent_limit = concurrent_limit

        self.queue: list[Future[R]] = []

        # We hold reference to this because we will wrap it with some
        # utility to handle tracebacks and possible pynisher
        self._original_function = function

        # We wrap the function such that when an error occurs, it's
        # traceback is attached to the message. This is because we
        # can't retrieve the traceback from an exception in another
        # process.
        self.function = exception_wrap(function)

        # If any of our limits is set, we need to wrap it in Pynisher
        # to enfore these limits.
        if any(
            limit is not None
            for limit in (memory_limit, cpu_time_limit, wall_time_limit)
        ):
            self.function = Pynisher(
                self.function,
                memory=memory_limit,
                cpu_time=cpu_time_limit,
                wall_time=wall_time_limit,
                terminate_child_processes=True,
            )

        # Pynisher limits
        self.memory_limit = memory_limit
        self.cpu_time_limit = cpu_time_limit
        self.wall_time_limit = wall_time_limit
        self.n_called = 0

        # Set up subscription methods to events
        self.on_submitted: Subscriber[Future[R]]
        self.on_submitted = self.subscriber(self.SUBMITTED)

        self.on_done: Subscriber[Future[R]]
        self.on_done = self.subscriber(self.DONE)

        self.on_cancelled: Subscriber[Future[R]]
        self.on_cancelled = self.subscriber(self.CANCELLED)

        self.on_returned: Subscriber[Future[R], R]
        self.on_returned = self.subscriber(self.RETURNED)

        self.on_exception: Subscriber[Future[R], BaseException]
        self.on_exception = self.subscriber(self.EXCEPTION)

        self.on_timeout: Subscriber[Future[R], BaseException]
        self.on_timeout = self.subscriber(self.TIMEOUT)

        self.on_memory_limit: Subscriber[Future[R], BaseException]
        self.on_memory_limit = self.subscriber(self.MEMORY_LIMIT_REACHED)

        self.on_cpu_time_limit_reached: Subscriber[Future[R], BaseException]
        self.on_cputime_limit_reached = self.subscriber(self.CPU_TIME_LIMIT_REACHED)

        self.on_walltime_limit_reached: Subscriber[Future[R], BaseException]
        self.on_walltime_limit_reached = self.subscriber(self.WALL_TIME_LIMIT_REACHED)

        self.on_call_limit_reached: Subscriber[P]
        self.on_call_limit_reached = self.subscriber(self.CALL_LIMIT_REACHED)

        self.on_concurrent_limit_reached: Subscriber[P]
        self.on_concurrent_limit_reached = self.subscriber(
            self.CONCURRENT_LIMIT_REACHED
        )

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

    @classmethod
    def events(cls) -> list[Event]:
        """Return the events that this class emits."""
        inherited_attrs = chain.from_iterable(vars(cls).values() for cls in cls.__mro__)
        return [attr for attr in inherited_attrs if isinstance(attr, Event)]

    def subscriber(self, event: Event[P2]) -> Subscriber[P2]:
        """Return an object that can be used to subscribe to an event for this task.

        Args:
            event: The event to subscribe to.

        Returns:
            A callable object that can be used to subscribe to events.
        """
        _event = (event, self.name)
        return self.scheduler.event_manager.subscriber(_event)

    def emit(self, event: Event, *args: Any, **kwargs: Any) -> None:
        """Emit an event.

        Args:
            event: The event to emit.
            *args: The positional arguments to pass to the event handlers.
            **kwargs: The keyword arguments to pass to the event handlers.
        """
        task_event = (event, self.name)
        self.scheduler.event_manager.emit(task_event, *args, **kwargs)

    def forward_event(self, frm: Hashable, to: Hashable) -> None:
        """Forward an event to another event.

        Args:
            frm: The event to forward.
            to: The event to forward to.
        """
        self.scheduler.event_manager.forward(frm, to)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Future[R] | None:
        """Dispatch this task.

        !!! note
            If `task.call_limit` was set and this limit was reached, the call
            will have no effect and nothing well be dispatched. Only a
            debug message will be logged. You can use `task.n_called`
            and `task.call_limit` to check if the limit was reached.

        Args:
            *args: The positional arguments to pass to the task.
            **kwargs: The keyword arguments to call the task with.

        Returns:
            The future of the task, or `None` if the limit was reached.
        """
        if self.call_limit and self.n_called >= self.call_limit:
            self.emit(self.CALL_LIMIT_REACHED, *args, **kwargs)
            return None

        n_running = sum(1 for f in self.queue if not f.done())
        if self.concurrent_limit is not None and n_running >= self.concurrent_limit:
            self.emit(self.CONCURRENT_LIMIT_REACHED, *args, **kwargs)
            return None

        self.n_called += 1
        future = self.scheduler._submit(self.function, *args, **kwargs)
        self.queue.append(future)

        # Process the task once it's completed
        future.add_done_callback(self._process_future)

        # We actuall have the function wrapped in something will
        # attach tracebacks to errors, so we need to get the
        # original function name.
        msg = (
            f"Submitted {self} with "
            f"{callstring(self._original_function, *args, **kwargs)}"
        )
        logger.debug(msg)
        self.emit(Task.SUBMITTED, future)

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
            self.emit(Task.EXCEPTION, future, exception)

            # If it was a limiting exception, emit it
            if isinstance(exception, Pynisher.TimeoutException):
                self.emit(Task.TIMEOUT, future, exception)
            if isinstance(exception, Pynisher.WallTimeoutException):
                self.emit(Task.WALL_TIME_LIMIT_REACHED, future, exception)
            if isinstance(exception, Pynisher.MemoryLimitException):
                self.emit(Task.MEMORY_LIMIT_REACHED, future, exception)
            if isinstance(exception, Pynisher.CpuTimeoutException):
                self.emit(Task.CPU_TIME_LIMIT_REACHED, future, exception)
        else:
            result = future.result()
            self.emit(Task.RETURNED, future, result)

    def __repr__(self) -> str:
        return f"Task(name={self.name})"
