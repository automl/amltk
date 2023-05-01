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
from typing import TYPE_CHECKING, Any, Callable, Generic, Hashable, TypeVar
from typing_extensions import ParamSpec

from pynisher import Pynisher

from byop.events import Event, Subscriber
from byop.exceptions import exception_wrap
from byop.functional import callstring, funcname

if TYPE_CHECKING:
    from concurrent.futures import Future

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
        n_called: How many times this task has been called.
        call_limit: How many times this task can be run. Defaults to `None`task
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

        self.on_f_returned: Subscriber[Future[R], R]
        self.on_f_returned = self.subscriber(self.F_RETURNED)

        self.on_f_exception: Subscriber[Future[R], BaseException]
        self.on_f_exception = self.subscriber(self.F_EXCEPTION)

        self.on_returned: Subscriber[R]
        self.on_returned = self.subscriber(self.RETURNED)

        self.on_exception: Subscriber[BaseException]
        self.on_exception = self.subscriber(self.EXCEPTION)

        self.on_timeout: Subscriber[Future[R], BaseException]
        self.on_timeout = self.subscriber(self.TIMEOUT)

        self.on_memory_limit_reached: Subscriber[Future[R], BaseException]
        self.on_memory_limit_reached = self.subscriber(self.MEMORY_LIMIT_REACHED)

        self.on_cpu_time_limit_reached: Subscriber[Future[R], BaseException]
        self.on_cputime_limit_reached = self.subscriber(self.CPU_TIME_LIMIT_REACHED)

        self.on_walltime_limit_reached: Subscriber[Future[R], BaseException]
        self.on_walltime_limit_reached = self.subscriber(self.WALL_TIME_LIMIT_REACHED)

        self.on_call_limit_reached: Subscriber[P]
        self.on_call_limit_reached = self.subscriber(self.CALL_LIMIT_REACHED)

        self.on_concurrent_limit_reached: Subscriber[P]
        self.on_concurrent_limit_reached = self.subscriber(
            self.CONCURRENT_LIMIT_REACHED,
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

        if (
            self.concurrent_limit is not None
            and self.n_running >= self.concurrent_limit
        ):
            self.emit(self.CONCURRENT_LIMIT_REACHED, *args, **kwargs)
            return None

        self.n_called += 1
        future = self.scheduler.submit(self.function, *args, **kwargs)
        if future is None:
            msg = (
                f"Task {callstring(self.function, *args, **kwargs)} was not"
                " able to be submitted. The scheduler is likely already finished."
            )
            logger.info(msg)
            return None

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
            emissions: dict = {
                Task.F_EXCEPTION: ((future, exception), None),
                Task.EXCEPTION: ((exception,), None),
            }

            # If it was a limiting exception, emit it
            if isinstance(exception, Pynisher.WallTimeoutException):
                emissions.update(
                    {
                        Task.TIMEOUT: ((future, exception), None),
                        Task.WALL_TIME_LIMIT_REACHED: ((future, exception), None),
                    },
                )
            elif isinstance(exception, Pynisher.MemoryLimitException):
                emissions.update(
                    {
                        Task.MEMORY_LIMIT_REACHED: ((future, exception), None),
                    },
                )
            elif isinstance(exception, Pynisher.CpuTimeoutException):
                emissions.update(
                    {
                        Task.TIMEOUT: ((future, exception), None),
                        Task.CPU_TIME_LIMIT_REACHED: ((future, exception), None),
                    },
                )

            self.emit_many(emissions)  # type: ignore
        else:
            result = future.result()
            self.emit_many(
                {
                    Task.F_RETURNED: ((future, result), None),
                    Task.RETURNED: ((result,), None),
                },
            )

    def __repr__(self) -> str:
        return f"Task(name={self.name})"
