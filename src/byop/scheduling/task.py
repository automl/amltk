"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.

There is also the [`CommTask`][byop.scheduling.comm_task.CommTask] which can
be used for communication between the task and the main process.
"""

from __future__ import annotations

from asyncio import Future
from collections import Counter
from dataclasses import dataclass, field
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

from pynisher import Pynisher
from typing_extensions import Self

from byop.exceptions import exception_wrap
from byop.functional import funcname
from byop.scheduling.events import TaskEvent
from byop.types import CallbackName, TaskName, TaskParams, TaskReturn

if TYPE_CHECKING:
    from byop.scheduling.scheduler import Scheduler

logger = logging.getLogger(__name__)


P = ParamSpec("P")
R = TypeVar("R")


class Task(Generic[TaskParams, TaskReturn]):
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

    def __init__(  # noqa: PLR0913
        self,
        name: TaskName,
        function: Callable[TaskParams, TaskReturn],
        scheduler: Scheduler,
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
        self.function: Callable[TaskParams, TaskReturn]

        self.name = name
        self.scheduler = scheduler
        self.call_limit = call_limit
        self.concurrent_limit = concurrent_limit

        # We wrap the function such that when an error occurs, it's
        # traceback is attached to the message. This is because we
        # can't retrieve the traceback from an exception in another
        # process.
        self.function = exception_wrap(function)

        # Pynisher limits
        self.memory_limit = memory_limit
        self.cpu_time_limit = cpu_time_limit
        self.wall_time_limit = wall_time_limit

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

        self.n_called = 0

    def running(self) -> bool:
        """Check if this task is currently running.

        Returns:
            bool: True if the task is currently running.
        """
        return any(future.running() for future in self.futures())

    def running_count(self) -> int:
        """Get the number of times this task is currently running.

        Returns:
            int: The number of this task that is currently running.
        """
        return sum(future.running() for future in self.futures())

    def futures(self) -> list[TaskFuture[TaskParams, TaskReturn]]:
        """Get the futures for this task.

        Returns:
            list[TaskFuture]: A list of futures for this task.
        """
        # NOTE: This could technically fail if there is multiple tasks
        # with different function definitions but the same name.
        futures = cast(
            list[TaskFuture[TaskParams, TaskReturn]],
            self.scheduler.task_futures.get(self.name, []),
        )
        return futures  # noqa: RET504

    @property
    def counts(self) -> Counter[TaskEvent]:
        """Get the number of event counts for this task.

        Returns:
            Counter[TaskEvent]: A counter of the number of times each event
                has been emitted.
        """
        counts = {
            event: self.scheduler.event_manager.counts[(self.name, event)]
            for event in iter(TaskEvent)
        }
        return Counter(counts)

    @overload
    def on(
        self,
        event: Literal[TaskEvent.SUBMITTED, TaskEvent.DONE, TaskEvent.CANCELLED],
        callback: Callable[[TaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.EXCEPTION],
        callback: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.RETURNED],
        callback: Callable[[TaskReturn], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskEvent,
        callback: Callable,
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task emits an event.

        ???+ note "See the more specific versions of this method for more"
            * [`on_submitted`][byop.scheduling.task.Task.on_submitted]
            * [`on_done`][byop.scheduling.task.Task.on_done]
            * [`on_cancelled`][byop.scheduling.task.Task.on_cancelled]
            * [`on_return`][byop.scheduling.task.Task.on_return]
            * [`on_noreturn`][byop.scheduling.task.Task.on_noreturn]

        Args:
            event: The event to listen to.
            callback: The callback to call.
            name: A specific name to give this callback.
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.
        """
        if name is None:
            if isinstance(callback, Task):  # noqa: SIM108
                name = callback.name
            else:
                name = funcname(callback)
            name = f"on-{self.name}-{event}-{name}"

        pred = None if when is None else (lambda counts=self.counts: when(counts))
        _event = (self.name, event)
        self.scheduler.event_manager.on(_event, callback, name=name, when=pred)
        return self

    def on_submitted(
        self,
        callback: Callable[[TaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task is submitted.

        Args:
            callback: The callback to call, which must accept this tasks
                TaskFuture as its only argument.
            name: A specific name to give this callback.
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.

        Returns:
            The task itself.
        """
        return self.on(TaskEvent.SUBMITTED, callback, name=name, when=when)

    def on_done(
        self,
        callback: Callable[[TaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task is done.

        Args:
            callback: The callback to call, which must accept this tasks
                TaskFuture as its only argument.
            name: A specific name to give this callback.
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.

        Returns:
            The task itself.
        """
        return self.on(TaskEvent.DONE, callback, name=name, when=when)

    def on_cancelled(
        self,
        callback: Callable[[TaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task is cancelled.

        Args:
            callback: The callback to call, which must accept this tasks
                TaskFuture as its only argument.
            name: A specific name to give this callback.
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.

        Returns:
            The task itself.
        """
        return self.on(TaskEvent.CANCELLED, callback, name=name, when=when)

    def on_return(
        self,
        callback: Callable[[TaskReturn], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task emits a success event.

        Args:
            callback: The callback to call, which must accept the return value
                of the task as its only argument.
            name: A specific name to give the callback
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.
        """
        return self.on(TaskEvent.RETURNED, callback, name=name, when=when)

    def on_exception(
        self,
        callback: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task emits a success event.

        Args:
            callback: The callback to call, which must accept the return value
                of the task as its only argument.
            name: A specific name to give this callback.
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.
        """
        return self.on(TaskEvent.EXCEPTION, callback, name=name, when=when)

    def __call__(
        self: Self,
        *args: TaskParams.args,
        **kwargs: TaskParams.kwargs,
    ) -> TaskFuture[TaskParams, TaskReturn] | None:
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
            msg = (
                f"Task {self.name} has been called {self.n_called} times,"
                f" reaching its call_limit {self.call_limit}."
            )
            logger.debug(msg)
            return None

        running_count = self.running_count()
        if self.concurrent_limit is not None and running_count >= self.concurrent_limit:
            msg = (
                f"Task {self.name} has {running_count} running instances,"
                f" reaching its concurrent_limit {self.concurrent_limit}."
            )
            logger.debug(msg)
            return None

        self.n_called += 1
        return self.scheduler._dispatch(self, *args, **kwargs)

    event: ClassVar = TaskEvent
    """The possible events a task can emit
    See [`TaskEvent`][byop.scheduling.events.TaskEvent] for more details.
    """

    SUBMITTED: ClassVar = TaskEvent.SUBMITTED
    """Event triggered when the task has been submitted.
    See [`TaskEvent.SUBMITTED`][byop.scheduling.events.TaskEvent.SUBMITTED].
    """

    DONE: ClassVar = TaskEvent.DONE
    """Event triggered when the task is done.
    See [`TaskEvent.DONE`][byop.scheduling.events.TaskEvent.DONE]
    """

    RETURNED: ClassVar = TaskEvent.RETURNED
    """Event triggered when the task has successfully returned a value.
    See [`TaskEvent.RETURNED`][byop.scheduling.events.TaskEvent.RETURNED]
    """

    EXCEPTION: ClassVar = TaskEvent.EXCEPTION
    """Event triggered when the task has errored.
    See [`TaskEvent.EXCEPTION`][byop.scheduling.events.TaskEvent.EXCEPTION]
    """

    CANCELLED: ClassVar = TaskEvent.CANCELLED
    """Event triggered when the task has been cancelled.
    See [`TaskEvent.CANCELLED`][byop.scheduling.events.TaskEvent.CANCELLED]
    """


@dataclass(frozen=True)
class TaskFuture(Generic[TaskParams, TaskReturn]):
    """A thin wrapper for a future with a name and reference to the task description.

    Attributes:
        desc: The task associated with this future.
        future: The future associated with this task.
    """

    desc: Task[TaskParams, TaskReturn]
    future: Future[TaskReturn] = field(repr=False)

    @property
    def name(self) -> TaskName:
        """The name of the task."""
        return self.desc.name

    @property
    def result(self) -> TaskReturn:
        """Get the result of the task."""
        return self.future.result()

    @property
    def exception(self) -> BaseException | None:
        """Get the exception of the task."""
        return self.future.exception()

    def has_result(self) -> bool:
        """Check if the task has a result."""
        return self.future.done() and not self.future.cancelled()

    def cancel(self) -> None:
        """Cancel the task."""
        self.future.cancel()

    def done(self) -> bool:
        """Check if the task is done."""
        return self.future.done()

    def cancelled(self) -> bool:
        """Check if the task is cancelled."""
        return self.future.cancelled()

    def running(self) -> bool:
        """Check if the task is running."""
        return not self.future.done()
