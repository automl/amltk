"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.
"""
from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterable,
    TypeVar,
)
from typing_extensions import ParamSpec, Self
from uuid import uuid4 as uuid

from amltk.events import Emitter, Event, Subscriber
from amltk.functional import callstring, funcname
from amltk.scheduling.sequential_executor import SequentialExecutor

if TYPE_CHECKING:
    from amltk.scheduling.scheduler import Scheduler
    from amltk.scheduling.task_plugin import TaskPlugin

logger = logging.getLogger(__name__)


P = ParamSpec("P")
P2 = ParamSpec("P2")

R = TypeVar("R")
R2 = TypeVar("R2")
CallableT = TypeVar("CallableT", bound=Callable)


class Task(Generic[P, R], Emitter):
    """A task is a unit of work that can be scheduled by the scheduler.

    It is defined by its `name` and a `function` to call. Whenever a task
    has its `__call__` method called, the function will be dispatched to run
    by a [`Scheduler`][amltk.scheduling.scheduler.Scheduler].

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
        uuid: A unique identifier for this task.
        unique_ref: A unique reference to this task.
        plugins: The plugins to use for this task.
        function: The function of this task
        scheduler: The scheduler that this task is registered with.
        init_plugins: Whether to initialize the plugins or not.
        queue: The queue of futures for this task.
        on_f_submitted: An event that is emitted when a future is submitted to the
            scheduler. It will pass the future as the first argument with args and
            kwargs following.

            This is done before any callbacks are attached to the future.
            ```python
            @task.on_f_submitted
            def on_f_submitted(future: Future, *args, **kwargs):
                print(f"Future {future} was submitted with {args=} and {kwargs=}")
            ```

        on_submitted: Called when a task is submitted to the scheduler.
            ```python
            @task.on_submitted
            def on_submitted(future: Future):
                print(f"Future {future} was submitted")
            ```

        on_done: Called when a task is done running with a result or exception.
            ```python
            @task.on_done
            def on_done(future: Future):
                print(f"Future {future} is done")
            ```

        on_cancelled: Called when a task is cancelled.
            ```python
            @task.on_cancelled
            def on_cancelled(future: Future):
                print(f"Future {future} was cancelled")
            ```
        on_f_returned: Called when a task has successfully returned a value.
            Comes with Future
            ```python
            @task.on_f_returned
            def on_f_returned(future: Future, result: R):
                print(f"Future {future} returned {result}")
            ```

        on_returned: Called when a task has successfully returned a value.
            ```python
            @task.on_returned
            def on_returned(result: R):
                print(f"Task returned {result}")
            ```
        on_f_exception: Called when a task failed to return anything but an exception.
            Comes with Future
            ```python
            @task.on_f_exception
            def on_f_exception(future: Future, error: BaseException):
                print(f"Future {future} exceptioned {error}")
            ```
        on_exception: Called when a task failed to return anything but an exception.
            ```python
            @task.on_exception
            def on_exception(error: BaseException):
                print(f"Task exceptioned {error}")
            ```
    """

    F_SUBMITTED: Event[...] = Event("task-future-submitted")
    """An event that is emitted when a future is submitted to the scheduler.
    It will pass the future as the first argument with args and kwargs following.

    This is done before any callbacks are attached to the future.
    """

    SUBMITTED: Event[Future[R]] = Event("task-submitted")
    """A Task has been submitted to the scheduler."""

    DONE: Event[Future[R]] = Event("task-done")
    """A Task has finished running. It is either returned or exceptioned.

    Cancelled is treated differently.
    """

    CANCELLED: Event[Future[R]] = Event("task-cancelled")
    """A Task has been cancelled."""

    F_RETURNED: Event[Future[R], R] = Event("task-future-returned")
    """A Task has successfully returned a value. Comes with Future"""

    RETURNED: Event[R] = Event("task-returned")
    """A Task has successfully returned a value."""

    F_EXCEPTION: Event[Future[R], BaseException] = Event("task-future-exception")
    """A Task failed to return anything but an exception. Comes with Future"""

    EXCEPTION: Event[BaseException] = Event("task-exception")
    """A Task failed to return anything but an exception."""

    def __init__(
        self: Self,
        function: Callable[P, R],
        scheduler: Scheduler,
        *,
        name: str | None = None,
        stop_on: Event | Iterable[Event] | None = None,
        plugins: Iterable[TaskPlugin[P, R]] = (),
        init_plugins: bool = True,
    ) -> None:
        """Initialize a task.

        Args:
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            name: The name of the task.
            stop_on: An event to stop the scheduler on.
            plugins: The plugins to use for this task.
            init_plugins: Whether to initialize the plugins or not.
        """
        self.name = name if name is not None else funcname(function)
        self.uuid = str(uuid())
        self.unique_ref = f"{self.name}-{self.uuid}"

        super().__init__(event_manager=self.unique_ref)

        self.plugins = plugins
        self.function = function
        self.scheduler = scheduler
        self.init_plugins = init_plugins
        self.queue: list[Future[R]] = []

        # Set up subscription methods to events
        self.on_f_submitted = self.subscriber(self.F_SUBMITTED)

        self.on_submitted = self.subscriber(self.SUBMITTED)

        self.on_done = self.subscriber(self.DONE)

        self.on_cancelled = self.subscriber(self.CANCELLED)

        self.on_f_returned = self.subscriber(self.F_RETURNED)
        self.on_f_exception = self.subscriber(self.F_EXCEPTION)
        self.on_returned = self.subscriber(self.RETURNED)
        self.on_exception = self.subscriber(self.EXCEPTION)

        # Used to keep track of any events emitted out of this task
        self._emitted_events: set[Event] = set()

        if stop_on is not None:
            stop_on = {stop_on} if isinstance(stop_on, Event) else set(stop_on)
            for _event in stop_on:
                subscriber: Subscriber = self.subscriber(_event)
                callback = lambda *args, _event=_event, **kwargs: self.scheduler.stop(
                    *args,
                    stop_msg=(
                        f"Task {self.unique_ref} emitted {_event} and was told to stop",
                    ),
                    **kwargs,
                )
                subscriber(callback)

        self._stop_on = stop_on

        if init_plugins:
            for plugin in self.plugins:
                plugin.attach_task(self)

    def futures(self) -> list[Future[R]]:
        """Get the futures for this task.

        Returns:
            A list of futures for this task.
        """
        return self.queue

    def on(
        self,
        event: Event[P2],
        *,
        name: str | None = None,
        when: Callable[[], bool] | None = None,
        limit: int | None = None,
        repeat: int = 1,
        every: int | None = None,
    ) -> Callable[[Callable[P2, R]], Callable[P2, R]]:
        """Decorator to subscribe to an event.

        Args:
            event: The event to subscribe to.
            name: See [`EventManager.on()`][amltk.events.EventManager.on].
            when: See [`EventManager.on()`][amltk.events.EventManager.on].
            limit: See [`EventManager.on()`][amltk.events.EventManager.on].
            repeat: See [`EventManager.on()`][amltk.events.EventManager.on].
            every: See [`EventManager.on()`][amltk.events.EventManager.on].

        Returns:
            A decorator that can be used to subscribe to events.
        """

        def decorator(callback: Callable[P2, R]) -> Callable[P2, R]:
            self.subscriber(event)(
                callback,
                name=name,
                when=when,
                limit=limit,
                repeat=repeat,
                every=every,
            )
            return callback

        return decorator

    @property
    def n_running(self) -> int:
        """Get the number of futures for this task that are currently running."""
        return sum(1 for f in self.queue if not f.done())

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
            stop_on=self._stop_on,
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
                logger.info(
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
            logger.info(msg)
            return None

        self.queue.append(future)

        # To allow any subclasses time to react to the future before emitting
        # events or scheduling callbacks.
        self.emit(self.F_SUBMITTED, future, *args, **kwargs)

        # We have the function wrapped in something will
        # attach tracebacks to errors, so we need to get the
        # original function name.
        msg = f"Submitted {callstring(self.function, *args, **kwargs)} from {self}."
        logger.debug(msg)
        self.emit(self.SUBMITTED, future)

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
            self.emit(self.CANCELLED, future)
            return

        self.emit(self.DONE, future)

        exception = future.exception()
        if exception is not None:
            self.emit_many(
                {
                    self.F_EXCEPTION: ((future, exception), None),
                    self.EXCEPTION: ((exception,), None),
                },
            )
        else:
            result = future.result()
            self.emit_many(
                {
                    self.F_RETURNED: ((future, result), None),
                    self.RETURNED: ((result,), None),
                },
            )

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

    def __repr__(self) -> str:
        kwargs = {k: v for k, v in [("unique_ref", self.unique_ref)] if v is not None}
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"

    def batch(self, args: Iterable[Iterable]) -> Task.Batch[P, R]:
        """Create a batch of tasks.

        Please see [`Task.Batch`][amltk.Task.Batch] for more.

        ```python
        from amltk import Task, Scheduler

        def f(x: int, y: int) -> int:
            return x + y

        scheduler = Scheduler.with_processes(2)

        task = Task(f)
        batch = task.batch([(1, 1), (2, 2), (3, 3)])

        @scheduler.on_start
        def start():
            batch.submit()

        @batch.on_batch_returned
        def batch_returned(results: list[int]):
            print(results)
        ```

        Args:
            args: The iterable of arguments to pass to the task.

        Returns:
            A batch of tasks.
        """
        return self.Batch(self, args)

    class Batch(Generic[P2, R2], Emitter):
        """A batch of tasks.

        This is for workflows which relies on the results of multiple subtasks
        to run and return, emitting
        [`BATCH_RETURNED`][amltk.Task.Batch.BATCH_RETURNED] with
        the result of all subtasks when they are all complete.
        A Batch can only be submitted once. Please create a new batch if
        you need to create multiple batches of a task.

        ```python
        @batch.on_batch_returned
        def batch_successful(results: list[R]):
            print(results)
        ```

        If any of the subtasks fail or are cancelled, the above event will not
        be emitted. Instead, [`BATCH_DONE`][amltk.Task.Batch.BATCH_DONE] will be
        emitted once everything has either given a result, exception or been
        cancelled.

        ```python
        @batch.on_batch_done
        def batch_done(
            results: list[R],
            exceptions: list[BaseException],
            cancelled: list[Future[R]],
        ):
            print(results)
            print(exceptions)
            print(cancelled)
        ```

        !!! tip "Cancelling a batch"

            You can cancel a batch by calling [`cancel`][amltk.Task.Batch.cancel]
            on the batch. This will cancel all subtasks which have not yet
            completed.

            ```python
            @batch.on_any_exception
            def batch_cancel(exception: BaseException):
                batch.cancel()
            ```

        !!! warning "Cancellation and Executors"

            Depending on the [Executor][concurrent.futures.Executor] /
            [Scheduler][amltk.Scheduler] used, the effect of cancellation differs.
            With python's built in
            [ProcessPoolExecutor][concurrent.futures.ProcessPoolExecutor],
            the running tasks will continue running, while only pending tasks
            will be cancelled. If using `Dask`, the running tasks will be cancelled
            depending on th backend used.

        If something goes wrong while **submitting** the entire batch, e.g. one of the
        submissions fails, by default, the batch will enter a failed state and emit
        [`BATCH_FAILED`][amltk.Task.Batch.BATCH_FAILED]. You can control this
        with `#!python cancel_on_failed_submission=`.
        This will automatically [`cancel()`][amltk.Task.Batch.cancel] the batch.
        This only takes place while the batch is being submitted.

        ```python
        @batch.on_batch_failed
        def batch_failed():
            ...
        ```

        You can listen to individual events within a batch with the following:

        * [`ANY_SUBMITTED`][amltk.Task.Batch.ANY_SUBMITTED] - `.on_any_submitted`
        * [`ANY_DONE`][amltk.Task.Batch.ANY_DONE] - `.on_any_done`
        * [`ANY_RETURNED`][amltk.Task.Batch.ANY_RETURNED] - `.on_any_returned`
        * [`ANY_EXCEPTION`][amltk.Task.Batch.ANY_EXCEPTION] - `.on_any_exception`
        * [`ANY_CANCELLED`][amltk.Task.Batch.ANY_CANCELLED] - `.on_any_cancelled`

        !!! warning "When the submission fails"

            When the submission fails or is cancelled, the above individual
            events will not fire anymore.

        !!! warning "SequentialExecutor"

            Unfortunatly, the [`SequentialExecutor`][amltk.SequentialExecutor]
            does not support batches and a [`RuntimeError`][RuntimeError]
            will be raised if you try to submit a batch to it.

        Attributes:
            uuid: The unique identifier for this batch.
            name: The name of this batch.
            unique_ref: The unique reference for this batch.
            cancel_on_failed_submission: Whether to stop submitting tasks if
                one fails to submit.
            failed_submission: Set to `True` when the batch failed to submit
                in its entirety and `cancel_on_failed_submission=True`.
            cancel_triggered: Set to `True` when the batch is cancelled or failed
                to submit in its entirety and `cancel_on_failed_submission=True`.

                Activated with [`cancel()`][amltk.Task.Batch.cancel].
            task: The task to run in parallel.
            args: An iterable of iterables, i.e. `[(1, 2), (3, 4)]`, where the first
                task will be called with `f(1, 2)` and the second with `f(3, 4)`.
            on_batch_done: A [`Subscriber`][amltk.Subscriber] called
                when all subtasks have completed, raised an exception or were
                cancelled.
                ```python
                @batch.on_batch_done
                def on_batch_done(
                    results: list[R2],
                    failed: list[BaseException],
                    cancelled: list[Future[R2]],
                ) -> None:
                    ...
                ```
            on_batch_returned: A [`Subscriber`][amltk.Subscriber] called
                when all subtasks have completed successfully.
                ```python
                @batch.on_batch_returned
                def on_batch_returned(results: list[R2]) -> None:
                    ...
                ```
            on_batch_submitted: A [`Subscriber`][amltk.Subscriber] called
                when all subtasks have been submitted.
                ```python
                @batch.on_batch_submitted
                def on_batch_submitted() -> None:
                    ...
                ```
            on_batch_failed: A [`Subscriber`][amltk.Subscriber] called
                when the batch failed to submit in its entirety and
                `cancel_on_failed_submission=True`.
                ```python
                @batch.on_batch_failed
                def on_batch_failed(failed: list[BaseException]) -> None:
                    ...
                ```
            on_batch_cancelled: A [`Subscriber`][amltk.Subscriber] called
                when [`.cancel()`][amltk.Task.Batch.cancel] is called or when
                `cancel_on_failed_submission=True` and the batch failed to submit
                ```python
                @batch.on_batch_cancelled
                def on_batch_cancelled() -> None:
                    ...
                ```
            on_any_exception: A [`Subscriber`][amltk.Subscriber]
                called when any subtask raises an exception.
                ```python
                @batch.on_any_exception
                def on_any_exception(exception: BaseException) -> None:
                    ...
                ```
            on_any_cancelled: A [`Subscriber`][amltk.Subscriber]
                called when any subtask is cancelled.
                ```python
                @batch.on_any_cancelled
                def on_any_cancelled(future: Future[R]) -> None:
                    ...
                ```
            on_any_returned: A [`Subscriber`][amltk.Subscriber]
                called when any subtask completes successfully.
                ```python
                @batch.on_any_returned
                def on_any_returned(result: R) -> None:
                    ...
                ```
            on_any_submitted: A [`Subscriber`][amltk.Subscriber]
                called when any subtask is submitted.
                ```python
                @batch.on_any_submitted
                def on_any_submitted(future: Future[R]) -> None:
                    ...
                ```
            on_any_done: A [`Subscriber`][amltk.Subscriber]
                called when any subtask completes or raises an exception.
                ```python
                @batch.on_any_done
                def on_any_done(future: Future[R]) -> None:
                    ...
                ```
        """

        BATCH_DONE: Event[
            list[R2],
            list[BaseException],
            list[Future[R2]],
        ] = Event("BATCH_DONE")
        """Event emitted when all subtasks have completed, raised an exception or were
        cancelled.

        Will return 3 lists, those that have completed, those that crashed and those
        that were cancelled.
        """

        BATCH_RETURNED: Event[list[R2]] = Event("BATCH_RETURNED")
        """Event emitted when all subtasks have completed successfully.

        Will return the results of all subtasks in the order of the arguments given.
        """

        BATCH_SUBMITTED: Event[list[Future[R2]]] = Event("BATCH_SUBMITTED")
        """Event emitted when all subtasks have been submitted."""

        BATCH_CANCELLED: Event[[]] = Event("BATCH_CANCELLED")
        """Event emitted when the entire batch is cancelled.

        This is triggered with [`Task.Batch.cancel()`][amltk.Task.Batch.cancel] and
        when `cancel_on_failed_submission=True` and submission fails to go through.
        """

        BATCH_FAILED: Event[[]] = Event("BATCH_FAILED")
        """Event emitted when the batch failed to submit in its entirety and
        `cancel_on_failed_submission=True`.
        """

        ANY_SUBMITTED: Event[Future[R2]] = Event("ANY_SUBMITTED")
        """Event emitted when any subtask was submitted."""

        ANY_EXCEPTION: Event[BaseException] = Event("ANY_EXCEPTION")
        """Event emitted when any subtask raises an exception."""

        ANY_RETURNED: Event[R2] = Event("ANY_RETURNED")
        """Event emitted when any subtask returns successfully."""

        ANY_CANCELLED: Event[Future[R2]] = Event("ANY_CANCELLED")
        """Event emitted when any subtask is cancelled."""

        ANY_DONE: Event[Future[R2]] = Event("ANY_DONE")
        """Event emitted when any subtask is done, regardless of the outcome.

        Does not fire when canceled.
        """

        def __init__(
            self,
            task: Task[P2, R2],
            args: Iterable[Iterable],
            *,
            cancel_on_failed_submission: bool = True,
        ) -> None:
            """Create a new batch of tasks.

            Args:
                task: The task to run in parallel.

                    !!! note "Copied Task"

                        The task will be copied, so that events emitted by this
                        are seperate from the original task.

                args: The arguments to pass to the task. This should be an iterable
                    of iterables, i.e. `[(1, 2), (3, 4)]`, where the first task will
                    be called with `f(1, 2)` and the second with `f(3, 4)`.
                cancel_on_failed_submission: Whether to stop submitting tasks if one
                    fails to submit. This will attempt to cancel all submitted tasks
                    and prevent any subtask completion events being triggered. When this
                    occurs, [`BATCH_FAILED`][amltk.Task.Batch.BATCH_FAILED]
                    will be triggered which can be subscribed to with
                    `on_failed_submission()`. Defaults to `True`.
            """
            # OPTIM: We could use a set for non-ordered container lookup,
            # as we do for emitting the ANY_* events.
            # For now we will just use `._submitted` and assume
            # that it is not too slow.

            # NOTE: SequentialExecutor strikes again!
            # Unfortunatly, we can't submit everything at once, as
            # the SequentialExecutor will only run one task at a time.
            # This means the BATCH_DONE will trigger before having all
            # tasks submitted.
            if isinstance(task.scheduler.executor, SequentialExecutor):
                raise ValueError(
                    "Cannot run a batch of tasks with a SequentialExecutor.",
                )

            self.uuid = str(uuid())
            self.name = f"batch-{task.name}"
            self.unique_ref = f"{self.name}-{self.uuid}"
            self.cancel_on_failed_submission = cancel_on_failed_submission
            self.failed_submission = False
            self.cancel_triggered = False

            super().__init__(event_manager=f"Batch-{self.unique_ref}")

            # NOTE: We need to copy the task to prevent events from the original
            # task from being mixed up with this particular batch
            self.task = task.copy()
            self.args = args
            self.on_batch_done = self.subscriber(self.BATCH_DONE)
            self.on_batch_returned = self.subscriber(self.BATCH_RETURNED)
            self.on_batch_submitted = self.subscriber(self.BATCH_SUBMITTED)
            self.on_batch_failed = self.subscriber(self.BATCH_FAILED)
            self.on_batch_cancelled = self.subscriber(self.BATCH_CANCELLED)
            self.on_any_exception = self.subscriber(self.ANY_EXCEPTION)
            self.on_any_cancelled = self.subscriber(self.ANY_CANCELLED)
            self.on_any_returned = self.subscriber(self.ANY_RETURNED)
            self.on_any_submitted = self.subscriber(self.ANY_SUBMITTED)
            self.on_any_done = self.subscriber(self.ANY_DONE)
            self._submitted: list[Future[R2]] = []
            self._results: dict[Future[R2], R2] = {}
            self._exceptions: dict[Future[R2], BaseException] = {}
            self._cancelled: list[Future[R2]] = []

            self.task.on_done(self._on_done)
            self.task.on_f_returned(self._on_returned)
            self.task.on_f_exception(self._on_exception)
            self.task.on_cancelled(self._on_cancelled)

        def _on_returned(self, future: Future[R2], result: R2) -> None:
            if self.failed_submission or self.cancel_triggered:
                logger.debug(f"Ignoring due to cancelled/failed submission of {self}.")
                return

            self._results[future] = result
            self.emit(self.ANY_RETURNED, result)
            self._maybe_finish()

        def _on_exception(self, future: Future[R2], exception: BaseException) -> None:
            if self.failed_submission or self.cancel_triggered:
                logger.debug(f"Ignoring due to cancelled/failed submission of {self}.")
                return

            self._exceptions[future] = exception
            self.emit(self.ANY_EXCEPTION, exception)
            self._maybe_finish()

        def _on_cancelled(self, future: Future[R2]) -> None:
            if self.failed_submission or self.cancel_triggered:
                logger.debug(f"Ignoring due to cancelled/failed submission of {self}.")
                return

            self._cancelled.append(future)
            self.emit(self.ANY_CANCELLED, future)
            self._maybe_finish()

        def _on_done(self, future: Future[R2]) -> None:
            if self.failed_submission or self.cancel_triggered:
                logger.debug(f"Ignoring due to cancelled/failed submission of {self}.")
                return

            self.emit(self.ANY_DONE, future)

        def cancel(self) -> None:
            """Cancel all subtasks."""
            # Already been cancelled, ignore this call
            if self.cancel_triggered:
                return

            self.cancel_triggered = True
            for future in self._submitted:
                future.cancel()

            self.emit(self.BATCH_CANCELLED)

        def submit(self) -> list[Future[R2]]:
            """Submit the batch of tasks.

            Returns:
                A list of futures for each subtask.
            """
            return self.__call__()

        def __call__(self) -> list[Future[R2]]:
            """Submit the batch of tasks.

            Returns:
                A list of futures for each subtask.
            """
            if any(self._submitted):
                raise RuntimeError(f"Cannot submit {self} more than once.")

            for _args in self.args:
                if self.cancel_triggered:
                    continue
                if (future := self.task.submit(*_args)) is not None:  # type: ignore
                    self._submitted.append(future)
                    self.emit(self.ANY_SUBMITTED, future)
                else:
                    logger.info(f"Was not able to submit all tasks for {self}.")
                    if self.cancel_on_failed_submission:
                        logger.info(f"Stopping submission for {self}.")
                        self.failed_submission = True
                        self.cancel()

            if self.failed_submission:
                self.emit(self.BATCH_FAILED)
                return list(self._submitted)

            if self.cancel_triggered:
                # Event emitted already in cancel()
                return list(self._submitted)

            submitted = list(self._submitted)
            self.emit(self.BATCH_SUBMITTED, submitted)

            return submitted

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.task})"

        def _maybe_finish(self) -> None:
            """Emit results if finished, removing callbacks if so."""
            n_done = len(self._results) + len(self._exceptions) + len(self._cancelled)
            n_submitted = len(self._submitted)

            if n_done > n_submitted:
                logger.warning(
                    f"Batch {self} has more tasks done ({n_done}) than submitted"
                    f" ({n_submitted}). This should not happen.",
                )
            elif n_submitted == n_done:
                results, exceptions, cancelled = self._collect()
                if len(exceptions) == 0 and len(cancelled) == 0:
                    self.emit(self.BATCH_RETURNED, results)

                self.emit(self.BATCH_DONE, results, exceptions, cancelled)

        def _collect(self) -> tuple[list[R2], list[BaseException], list[Future[R2]]]:
            """Collect the results of the batch.

            Returns:
                A tuple of the results, exceptions and futures.
            """
            results = [
                result
                for future in self._submitted
                if (result := self._results.get(future)) is not None
            ]
            exceptions = [
                exception
                for future in self._submitted
                if (exception := self._exceptions.get(future)) is not None
            ]
            return results, exceptions, self._cancelled
