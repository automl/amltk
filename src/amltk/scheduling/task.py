"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.
"""
from __future__ import annotations

import logging
from asyncio import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    TypeVar,
    overload,
)
from typing_extensions import Concatenate, ParamSpec, Self, override
from uuid import uuid4 as uuid

from amltk.events import Emitter, Event, Subscriber
from amltk.functional import callstring, funcname

if TYPE_CHECKING:
    from amltk.scheduling.scheduler import Scheduler
    from amltk.scheduling.task_plugin import TaskPlugin

logger = logging.getLogger(__name__)


P = ParamSpec("P")
P2 = ParamSpec("P2")

R = TypeVar("R")
R2 = TypeVar("R2")
CallableT = TypeVar("CallableT", bound=Callable)


class Task(Generic[P, R]):
    """A task is a unit of work that can be scheduled by the scheduler.

    It is defined by its `name` and a `function` to call. Whenever a task
    has its `__call__` method called, the function will be dispatched to run
    by a [`Scheduler`][amltk.scheduling.scheduler.Scheduler].

    The scheduler will emit specific events
    to this task which look like `(task.name, TaskEvent)`.

    To interact with the results of these tasks, you must subscribe to to these
    events and provide callbacks.

    ```python
    from amltk import Task, Scheduler

    # Define some function to run
    def f(x: int) -> int:
        return x * 2

    # And a scheduler to run it on
    scheduler = Scheduler.with_processes(2)

    # Create the task object, the type anotation Task[[int], int] isn't required
    my_task = Task(f, scheduler)

    # Subscribe to events
    @my_task.on_returned
    def print_result(future: Future[int], result: int):
        print(f"Future {future} returned {result}")

    @my_task.on_exception
    def print_exception(future: Future[int], result: int):
        print(error)
    ```

    If providing `plugins=` to the task, these may add new events that will be emitted
    from the task. Do listen to these events, you must use the `on` method. Please
    see their respective documentation.

    ```python
    from amltk import Scheduler, Task
    from amltk.scheduling import CallLimiter


    def f() -> None:
        print("task ran")


    scheduler = Scheduler.with_processes(1)
    task = Task(f, scheduler, plugins=[CallLimiter(max_calls=1)])

    @scheduler.on_start(repeat=3)
    def start():
        task()

    @task.on(CallLimiter.CALL_LIMIT_REACHED)
    def on_limit_reached(task: Task, *args, **kwargs):
        print(f"Task {task} reached its call limit with {args=} and {kwargs=}")

    scheduler.run()
    ```
    """

    name: str
    """The name of the task."""
    uuid: str
    """A unique identifier for this task."""
    unique_ref: str
    """A unique reference to this task."""
    plugins: list[TaskPlugin]
    """The plugins to use for this task."""
    function: Callable[P, R]
    """The function of this task"""
    scheduler: Scheduler
    """The scheduler that this task is registered with."""
    init_plugins: bool
    """Whether to initialize the plugins or not."""
    queue: list[Future[R]]
    """The queue of futures for this task."""
    emitter: Emitter
    """The emitter for events of this task."""

    on_submitted: Subscriber[Concatenate[Future[R], P]]
    """An event that is emitted when a future is submitted to the
    scheduler. It will pass the future as the first argument with args and
    kwargs following.

    This is done before any callbacks are attached to the future.
    ```python
    @task.on_submitted
    def on_submitted(future: Future[R], *args, **kwargs):
        print(f"Future {future} was submitted with {args=} and {kwargs=}")
    ```
    """
    on_done: Subscriber[Future[R]]
    """Called when a task is done running with a result or exception.
    ```python
    @task.on_done
    def on_done(future: Future[R]):
        print(f"Future {future} is done")
    ```
    """
    on_cancelled: Subscriber[Future[R]]
    """Called when a task is cancelled.
    ```python
    @task.on_cancelled
    def on_cancelled(future: Future[R]):
        print(f"Future {future} was cancelled")
    ```
    """
    on_returned: Subscriber[Future[R], R]
    """Called when a task has successfully returned a value.
    Comes with Future
    ```python
    @task.on_returned
    def on_returned(future: Future[R], result: R):
        print(f"Future {future} returned {result}")
    ```
    """
    on_exception: Subscriber[Future[R], BaseException]
    """Called when a task failed to return anything but an exception.
    Comes with Future
    ```python
    @task.on_exception
    def on_exception(future: Future[R], error: BaseException):
        print(f"Future {future} exceptioned {error}")
    ```
    """

    SUBMITTED: Event[Concatenate[Future[R], P]] = Event("task-submitted")
    DONE: Event[Future[R]] = Event("task-done")

    CANCELLED: Event[Future[R]] = Event("task-cancelled")
    RETURNED: Event[Future[R], R] = Event("task-returned")
    EXCEPTION: Event[Future[R], BaseException] = Event("task-exception")

    def __init__(
        self: Self,
        function: Callable[P, R],
        scheduler: Scheduler,
        *,
        name: str | None = None,
        plugins: Iterable[TaskPlugin] = (),
        init_plugins: bool = True,
    ) -> None:
        """Initialize a task.

        Args:
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            name: The name of the task.
            plugins: The plugins to use for this task.
            init_plugins: Whether to initialize the plugins or not.
        """
        super().__init__()
        self.name = name if name is not None else funcname(function)
        self.unique_ref = f"{self.name}-{uuid()}"

        self.emitter = Emitter(self.unique_ref)
        self.event_counts = self.emitter.event_counts
        self.plugins: list[TaskPlugin] = list(plugins)
        self.function: Callable[P, R] = function
        self.scheduler: Scheduler = scheduler
        self.init_plugins: bool = init_plugins
        self.queue: list[Future[R]] = []

        # Set up subscription methods to events
        self.on_submitted = self.emitter.subscriber(self.SUBMITTED)
        self.on_done = self.emitter.subscriber(self.DONE)
        self.on_returned = self.emitter.subscriber(self.RETURNED)
        self.on_exception = self.emitter.subscriber(self.EXCEPTION)
        self.on_cancelled = self.emitter.subscriber(self.CANCELLED)

        if init_plugins:
            for plugin in self.plugins:
                plugin.attach_task(self)

    def futures(self) -> list[Future[R]]:
        """Get the futures for this task.

        Returns:
            A list of futures for this task.
        """
        return self.queue

    @overload
    def on(
        self,
        event: Event[P2],
        callback: None = None,
        *,
        name: str | None = ...,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int | None = ...,
    ) -> Subscriber[P2]:
        ...

    @overload
    def on(
        self,
        event: Event[P2],
        callback: Callable[P2, Any] | Iterable[Callable[P2, Any]],
        *,
        name: str | None = ...,
        when: Callable[[], bool] | None = ...,
        limit: int | None = ...,
        repeat: int = ...,
        every: int | None = ...,
    ) -> Subscriber[P2]:
        ...

    def on(
        self,
        event: Event[P2],
        callback: Callable[P2, Any] | Iterable[Callable[P2, Any]] | None = None,
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> Subscriber[P2] | None:
        """Subscribe to an event.

        Args:
            event: The event to subscribe to.
            callback: The callback to call when the event is emitted.
                If not specified, what is returned can be used as a decorator.
            name: The name of the subscriber.
            when: A predicate to determine whether to call the callback.
            limit: The number of times to call the callback.
            repeat: The number of times to repeat the subscription.
            every: The number of times to wait between repeats.

        Returns:
            The subscriber if no callback was provided, otherwise `None`.
        """
        subscriber = self.emitter.subscriber(
            event,
            name=name,
            when=when,
            limit=limit,
            repeat=repeat,
            every=every,
        )
        if callback is None:
            return subscriber

        subscriber(callback)
        return None

    @property
    def n_running(self) -> int:
        """Get the number of futures for this task that are currently running."""
        return sum(1 for f in self.queue if not f.done())

    def running(self) -> bool:
        """Check if this task has any futures that are currently running."""
        return self.n_running > 0

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Future[R] | None:
        """Dispatch this task.

        Args:
            *args: The positional arguments to pass to the task.
            **kwargs: The keyword arguments to call the task with.

        Returns:
            The future of the task, or `None` if the limit was reached.
        """
        return self.__call__(*args, **kwargs)

    def copy(self, *, init_plugins: bool = True) -> Self:
        """Create a copy of this task.

        Will use the same scheduler and function, but will have a different
        event manager such that any events listend to on the old task will
        **not** trigger with the copied task.

        Args:
            init_plugins: Whether to initialize the copied plugins on the copied
                task. Usually you will want to leave this as `True`.

        Returns:
            A copy of this task.
        """
        return self.__class__(
            self.function,
            self.scheduler,
            name=self.name,
            plugins=tuple(p.copy() for p in self.plugins),
            init_plugins=init_plugins,
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Future[R] | None:
        """Please see [`Task.submit()`][amltk.Task.submit]."""
        # Inform all plugins that the task is about to be called
        # They have chance to cancel submission based on their return
        # value.
        fn = self.function
        for plugin in self.plugins:
            items = plugin.pre_submit(fn, *args, **kwargs)
            if items is None:
                logger.debug(
                    f"Plugin '{plugin.name}' prevented {self} from being submitted"
                    f" with {callstring(self.function, *args, **kwargs)}",
                )
                return None

            fn, args, kwargs = items  # type: ignore

        future = self.scheduler.submit(fn, *args, **kwargs)

        if future is None:
            msg = (
                f"Task {callstring(self.function, *args, **kwargs)} was not"
                " able to be submitted. The scheduler is likely already finished."
            )
            logger.debug(msg)
            return None

        self.queue.append(future)

        # We have the function wrapped in something will
        # attach tracebacks to errors, so we need to get the
        # original function name.
        msg = f"Submitted {callstring(self.function, *args, **kwargs)} from {self}."
        logger.debug(msg)
        self.on_submitted.emit(future, *args, **kwargs)

        # Process the task once it's completed
        # NOTE: If the task is done super quickly or in the sequential mode,
        # this will immediatly call `self._process_future`.
        future.add_done_callback(self._process_future)
        return future

    def _process_future(self, future: Future[R]) -> None:
        try:
            self.queue.remove(future)
        except ValueError as e:
            raise ValueError(f"{future=} not found in task queue {self.queue=}") from e

        if future.cancelled():
            self.on_cancelled.emit(future)
            return

        self.on_done.emit(future)

        exception = future.exception()
        if exception is not None:
            self.on_exception.emit(future, exception)
        else:
            result = future.result()
            self.on_returned.emit(future, result)

    def attach_plugin(self, plugin: TaskPlugin) -> None:
        """Attach a plugin to this task.

        Args:
            plugin: The plugin to attach.
        """
        self.plugins.append(plugin)
        plugin.attach_task(self)

    def _when_future_from_submission(
        self,
        future: Future[R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Access the future before the callbacks for the future are registered.

        This is primarly to allow subclasses of Task to know of the future obtained
        from submitting to the Scheduler, **before** the callbacks are registered.
        This can be required in the case the task finishes super quickly, meaning
        callbacks are registered before the subtask can do anything. This also
        necessarily happens in a sequential execution Scheduler.
        """

    @override
    def __repr__(self) -> str:
        kwargs = {"unique_ref": self.unique_ref}
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
