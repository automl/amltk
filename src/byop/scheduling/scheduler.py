"""A scheduler which uses asyncio and an executor to run tasks concurrently.

It's primary use is to dispatch tasks to an executor and manage callbacks
for when they complete.
"""
from __future__ import annotations

import asyncio
from asyncio import Future
from collections import Counter, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
import logging
from multiprocessing.context import BaseContext
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Hashable,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)
from uuid import uuid4

from typing_extensions import Self

from byop.event_manager import EventManager
from byop.functional import funcname
from byop.scheduling.comm_task import Comm, CommTask, CommTaskFuture
from byop.scheduling.events import EventTypes, ExitCode, SchedulerEvent, TaskEvent
from byop.scheduling.task import Task, TaskFuture
from byop.scheduling.termination_strategies import termination_strategy
from byop.types import CallbackName, Msg, TaskName, TaskParams, TaskReturn

P = ParamSpec("P")
R = TypeVar("R")
TaskT = TypeVar("TaskT", bound=Task)
CommTaskT = TypeVar("CommTaskT", bound=CommTask)

logger = logging.getLogger(__name__)


class Scheduler:
    """A scheduler for submitting tasks to an Executor."""

    def __init__(
        self,
        executor: Executor,
        *,
        terminate: Callable[[Executor], None] | bool = True,
    ) -> None:
        """Initialize the scheduler.

        Note:
            As an `Executor` does not provide an interface to forcibly
            terminate workers, we provide `terminate_workers` as a custom
            strategy for cleaning up a provided executor. It is not possible
            to terminate running thread based workers, for example using
            `ThreadPoolExecutor` and any Executor using threads to spawn
            tasks will have to wait until all running tasks are finish
            before python can close.

        Args:
            executor: The dispatcher to use for submitting tasks.
            terminate: Whether to call shutdown on the executor when
                `run(..., wait=False)`. If True, the executor will be
                `shutdown(wait=False)` and we will attempt to terminate
                any workers of the executor. For some `Executors` this
                is enough, i.e. Dask, however for something like
                `ProcessPoolExecutor`, we will use `psutil` to kill
                its worker processes. If a callable, we will use this
                function for custom worker termination.
                If False, shutdown will not be called and the executor will
                remain active.

        """
        self.executor = executor

        self.terminate: Callable[[Executor], None] | None
        if terminate is True:
            self.terminate = termination_strategy(executor)
        else:
            self.terminate = terminate if callable(terminate) else None

        # An event managers which handles task status and calls callbacks
        # NOTE: Typing the event manager is a little complicated, so we
        # forego it for now. However it is possible
        self.event_manager: EventManager = EventManager(name="Scheduler")

        # The current state of things and references to them
        self.queue: dict[Future, TaskFuture] = {}

        # Futures indexed by the task name
        self.task_futures: dict[TaskName, list[TaskFuture]] = defaultdict(list)

        # This can be triggered either by `scheduler.stop` in a callback
        self._stop_event: asyncio.Event = asyncio.Event()

        # This is a condition to make sure monitoring the queue will wait
        # properly
        self._queue_has_items: asyncio.Event = asyncio.Event()

        # This is triggered when run is called
        self._running: asyncio.Event = asyncio.Event()

        # The currently open communcation with `dispatch_with_comm` workers
        self.communcations: dict[TaskName, asyncio.Task[None]] = {}

    @classmethod
    def with_processes(
        cls,
        max_workers: int | None = None,
        mp_context: BaseContext | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> Self:
        """Create a scheduler with a `ProcessPoolExecutor`.

        See [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor]
        for more details.
        """
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
        )
        return cls(executor=executor)

    def empty(self) -> bool:
        """Check if the scheduler is empty.

        Returns:
            True if there are no tasks in the queue.
        """
        return len(self.queue) == 0

    def running(self) -> bool:
        """Whether the scheduler is running and accepting tasks to dispatch.

        Returns:
            True if the scheduler is running and accepting tasks.
        """
        return self._running.is_set()

    @property
    def counts(self) -> dict[TaskEvent, int]:
        """The event counter.

        Useful for predicates, for example
        ```python
        from byop.scheduling import TaskEvent

        my_scheduler.on_task_finished(
            do_something,
            when=lambda sched: sched.counts[TaskEvent.FINISHED] > 10
        )
        ```
        """
        return self.event_manager.counts

    @overload
    def task(
        self,
        function: Callable[P, R],
        *,
        name: TaskName | None | Literal[True] = ...,
        limit: int | None = ...,
        comms: Literal[False] = False,
        task_type: None = None,
    ) -> Task[P, R]:
        ...

    @overload
    def task(
        self,
        function: Callable[Concatenate[Comm, P], R],
        *,
        name: TaskName | None | Literal[True] = ...,
        limit: int | None = ...,
        comms: Literal[True],
        task_type: None = None,
    ) -> CommTask[P, R]:
        ...

    @overload
    def task(
        self,
        function: Callable[Concatenate[Comm, P], R],
        *,
        name: TaskName | None | Literal[True] = ...,
        limit: int | None = ...,
        comms: Literal[True],
        task_type: type[CommTaskT],
    ) -> CommTaskT:
        ...

    @overload
    def task(
        self,
        function: Callable[P, Any],
        *,
        name: TaskName | None | Literal[True] = ...,
        limit: int | None = ...,
        comms: bool = ...,
        task_type: type[TaskT],
    ) -> TaskT:
        ...

    def task(  # type: ignore
        self,
        function: Callable[P, R] | Callable[Concatenate[Comm, P], R],
        *,
        name: TaskName | None | Literal[True] = None,
        limit: int | None = None,
        comms: bool = False,
        task_type: type[TaskT] | type[CommTaskT] | None = None,
    ) -> Task[P, R] | CommTask[P, R] | TaskT | CommTaskT:
        """Define a scheduler task.

        Args:
            function: The function to wrap up as a task.
            name: The name of the task, if None, the name of the function
                will be used. If True, a unique name will be generated.
            limit: The maximum number of times this task can be run.
            comms: Whether the function requires a `Comm` to `send`
                and `recv`.
            task_type: The custom type of task to create. Must extend
                from [`Task`][byop.scheduling.Task]
                or [`CommTask`][byop.scheduling.CommTask].
                If left as `None`, the default, a [`Task`][byop.scheduling.Task]
                will be created if `comms` is `False`, otherwise a
                [`CommTask`][byop.scheduling.CommTask] will be created.

        Returns:
            A Task which can be used to set up callbacks.
        """
        if name is None:
            name = funcname(function)
        elif name is True:
            name = str(uuid4())

        if task_type is not None:
            return task_type(
                function=function,
                name=name,
                limit=limit,
                scheduler=self,
            )

        task_cls = CommTask if comms else Task
        return task_cls(
            function=function,
            name=name,
            limit=limit,
            scheduler=self,
        )

    def _dispatch(
        self,
        task: Task[P, R] | CommTask[Concatenate[Comm, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> TaskFuture[P, R]:
        if not self._running:
            raise RuntimeError("Scheduler is not currently running")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            logger.error(
                "No async loop present, the scheduler `_dispatch` can only"
                " be called inside an event loop, as possible with `scheduler.run`"
            )
            raise e

        task_future: TaskFuture[P, R] | CommTaskFuture[Concatenate[Comm, P], R]
        if not isinstance(task, CommTask):
            scheduler_comm = None
            sync_future = self.executor.submit(task.function, *args, **kwargs)
            future = asyncio.wrap_future(sync_future, loop=loop)
            task_future = TaskFuture(future=future, desc=task)
        else:
            task = cast(CommTask[Concatenate[Comm, P], R], task)
            scheduler_comm, worker_comm = Comm.create(duplex=True)
            sync_future = self.executor.submit(
                task.function,
                *(worker_comm, *args),  # type: ignore
                **kwargs,
            )
            future = asyncio.wrap_future(sync_future, loop=loop)
            task_future = CommTaskFuture(future=future, desc=task, comm=scheduler_comm)

        future.add_done_callback(self._on_task_complete)

        if isinstance(task_future, CommTaskFuture):
            assert scheduler_comm is not None
            asyncio_task = asyncio.create_task(self._communicate(task_future))
            self.communcations[task_future.name] = asyncio_task

        # Place our future and task in the queue
        self.queue[future] = task_future
        self.task_futures[task.name].append(task_future)
        self._queue_has_items.set()

        # Emit the general submitted event and the task specific submitted event
        logger.debug(f"Submitted task {task}")
        self.event_manager.emit(TaskEvent.SUBMITTED, task)
        self.event_manager.emit((task.name, TaskEvent.SUBMITTED), task)
        return task_future

    def _on_task_complete(self, future: asyncio.Future) -> None:
        # Remove it fom the queue and get the wrapped task_future object
        task_future = self.queue.pop(future, None)
        if task_future is None:
            logger.warning(f"Task for {task_future} was not found in scheduler queue!")
            return

        # Try remove it from the task to futures lookup
        task_name = task_future.name
        if task_name not in self.task_futures:
            logger.warning(f"Task {task_name=} was not found {self.task_futures=}!")
        else:
            task_futures = self.task_futures[task_name]
            try:
                task_futures.remove(task_future)
            except ValueError:
                logger.warning(
                    f"Future {task_future} was not found in futures"
                    f" for {task_name=}!"
                )

        # NOTE: I have no reason to choose whether to emit the general
        # or task specific event first. If there is a compelling argument
        # to choose one over the other, please raise an issue dicussing it!
        # @eddiebergman
        if task_future.cancelled():
            logger.debug(f"Task {task_name} was cancelled")

            self.event_manager.emit(TaskEvent.CANCELLED, task_future)
            self.event_manager.emit((task_name, TaskEvent.CANCELLED), task_future)
            return

        exception = future.exception()
        result = future.result() if exception is None else None

        logger.debug(f"Task {task_future} finished")
        self.event_manager.emit(TaskEvent.DONE, task_future)
        self.event_manager.emit((task_name, TaskEvent.DONE), task_future)

        if exception is None:
            logger.debug(f"{task_future} completed successfully")
            self.event_manager.emit(TaskEvent.RETURNED, result)
            self.event_manager.emit((task_name, TaskEvent.RETURNED), result)
        else:
            logger.debug(f"{task_future} failed with Exception:`{exception}`")
            self.event_manager.emit(TaskEvent.NO_RETURN, exception)
            self.event_manager.emit((task_name, TaskEvent.NO_RETURN), exception)

        # If dealing with a comm task, get the async task that was in
        # charge of monitoring the pipes and cancel it, as well as close
        # remaining comms
        if isinstance(task_future, CommTaskFuture):
            async_communicate_task = self.communcations.pop(task_name, None)
            if async_communicate_task is None:
                msg = (
                    f"Task to communicate with {task_future} was not found"
                    " in the scheduler!"
                )
                logger.warning(msg)
            else:
                async_communicate_task.cancel()

            # Close the pipe if it hasn't been closed
            assert isinstance(task_future, CommTaskFuture)
            task_future.comm.close()

    async def _communicate(self, task: CommTaskFuture) -> None:
        """Communicate with the task.

        This is a coroutine that will run until the scheduler is stopped or
        the comms have finished.
        """
        while True:
            try:
                msg = await task.comm.as_async.recv()
                logger.debug(f"Worker {task.name}: receieved {msg}")
                if msg == TaskEvent.WAITING:
                    self.event_manager.emit(TaskEvent.WAITING)
                    self.event_manager.emit((task.name, TaskEvent.WAITING), task)
                else:
                    self.event_manager.emit(TaskEvent.UPDATE)
                    self.event_manager.emit((task.name, TaskEvent.UPDATE), task, msg)
            except EOFError:
                logger.debug(f"Worker {task.name}: closed connection")
                break

        # We are out of the loop, there's no way to communicate with
        # the worker anymore, close out and remove reference to this
        # task from the scheduler
        task.comm.close()

    async def _stop_when_queue_empty(self) -> None:
        """Stop the scheduler when the queue is empty."""
        while self.queue:
            await asyncio.wait(self.queue, return_when=asyncio.ALL_COMPLETED)

        logger.debug("Queue is empty, stopping scheduler")
        return

    async def _monitor_queue_empty(self) -> None:
        """Monitor for the queue being empty and trigger an event when it is."""
        while True:
            while self.queue:
                await asyncio.wait(self.queue, return_when=asyncio.ALL_COMPLETED)

            # Signal that the queue is now empty
            self._queue_has_items.clear()
            self.event_manager.emit(SchedulerEvent.EMPTY)

            # Wait for an item to be in the queue
            await self._queue_has_items.wait()
            logger.debug("Queue has been filled again")

    async def _stop_when_triggered(self) -> None:
        """Stop the scheduler when the stop event is set."""
        await self._stop_event.wait()

        logger.debug("Stop event triggered, stopping scheduler")
        return

    async def _run_scheduler(  # noqa: PLR0912, C901
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        self.executor.__enter__()

        # Declare we are running
        self._running.set()

        self.event_manager.emit(SchedulerEvent.STARTED)

        # Our stopping criterion of the scheduler
        stop_criterion: list[asyncio.Task] = []

        # Monitor for `stop` being triggered
        stop_triggered = asyncio.create_task(self._stop_when_triggered())
        stop_criterion.append(stop_triggered)

        # Monitor for the queue being empty
        if end_on_empty:
            queue_empty = asyncio.create_task(self._stop_when_queue_empty())
            stop_criterion.append(queue_empty)
            monitor_empty = None
        else:
            monitor_empty = asyncio.create_task(self._monitor_queue_empty())
            queue_empty = None

        # The timeout criterion is satisifed by the `timeout` arg
        await asyncio.wait(
            stop_criterion,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Determine the reason for stopping
        if stop_triggered.done():
            stop_reason = ExitCode.STOPPED
            self.event_manager.emit(SchedulerEvent.STOP)
        elif queue_empty and queue_empty.done():
            stop_reason = ExitCode.EXHAUSTED
        elif timeout is not None:
            logger.debug(f"Timeout of {timeout} reached for scheduler")
            stop_reason = ExitCode.TIMEOUT
            self.event_manager.emit(SchedulerEvent.TIMEOUT)
        else:
            logger.warning("Scheduler stopped for unknown reason!")
            stop_reason = ExitCode.UNKNOWN

        # Stop monitoring the queue to trigger an event
        if monitor_empty:
            monitor_empty.cancel()

        # Cancel the stopping criterion
        for stopping_criteria in stop_criterion:
            stopping_criteria.cancel()

        self.event_manager.emit(SchedulerEvent.FINISHING)

        if self.terminate:
            logger.debug(f"Shutting down scheduler executor with {wait=}")
            if wait:
                logger.debug("Waiting for jobs to finish in executor shutdown")

            self.executor.shutdown(wait=wait)

        # The scheduler is now refusing jobs
        self._running.clear()
        logger.debug("Scheduler has shutdown and declared as no longer running")

        # We do a manual `cancel_futures` here since it doesn't seem part of dask api
        if stop_reason in (ExitCode.TIMEOUT, ExitCode.STOPPED) and not wait:
            for future in self.queue:
                if not future.done():
                    future.cancel()
        else:
            logger.debug("Waiting for running tasks to finish and process...")
            queue_empty = asyncio.create_task(self._stop_when_queue_empty())
            await queue_empty

        # If we are meant to shut down the executor and terminate the workers
        # we should do so.
        if self.terminate:
            self.terminate(self.executor)

        self.event_manager.emit(SchedulerEvent.FINISHED)
        logger.info(f"Scheduler finished with status {stop_reason}")
        return stop_reason

    def run(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        """Run the scheduler.

        Args:
            timeout: The maximum time to run the scheduler for.
                Defaults to `None` which means no timeout and it
                will end once the queue becomes empty.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.

        Returns:
            The reason for the scheduler ending.

        Raises:
            RuntimeError: If the scheduler is already running.
        """
        if self.running():
            raise RuntimeError("Scheduler already seems to be running")

        logger.debug("Starting scheduler")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._run_scheduler(timeout=timeout, end_on_empty=end_on_empty, wait=wait)
        )

    async def async_run(
        self,
        initial: Task | list[Task] | None = None,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        """Async version of `run`.

        Args:
            initial: The initial tasks to run. Defaults to `None`
            timeout: The maximum time to run the scheduler for.
                Defaults to `None` which means no timeout.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.

        Returns:
            The reason for the scheduler ending.
        """
        if initial:
            if isinstance(initial, Task):
                initial = [initial]

            for task in initial:
                self.on(SchedulerEvent.STARTED, task, name=task.name)

        return await self._run_scheduler(
            timeout=timeout,
            end_on_empty=end_on_empty,
            wait=wait,
        )

    @overload
    def on(
        self,
        event: SchedulerEvent,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: (
            Literal[TaskEvent.SUBMITTED, TaskEvent.DONE, TaskEvent.CANCELLED]
            | tuple[
                TaskName,
                Literal[TaskEvent.SUBMITTED, TaskEvent.DONE, TaskEvent.CANCELLED],
            ]
        ),
        callback: (
            Callable[[TaskFuture[TaskParams, TaskReturn]], Any]
            | Task[[TaskFuture[TaskParams, TaskReturn]], Any]
        ),
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.RETURNED]
        | tuple[TaskName, Literal[TaskEvent.RETURNED]],
        callback: Callable[[TaskReturn], Any] | Task[[TaskReturn], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.NO_RETURN]
        | tuple[TaskName, Literal[TaskEvent.NO_RETURN]],
        callback: Callable[[BaseException], Any] | Task[[BaseException], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.UPDATE] | tuple[TaskName, Literal[TaskEvent.UPDATE]],
        callback: (
            Callable[[CommTaskFuture[TaskParams, TaskReturn], Msg], Any]
            | Task[[CommTaskFuture[TaskParams, TaskReturn], Msg], Any]
        ),
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.WAITING] | tuple[TaskName, Literal[TaskEvent.WAITING]],
        callback: (
            Callable[[CommTaskFuture[TaskParams, TaskReturn]], Any]
            | Task[[CommTaskFuture[TaskParams, TaskReturn]], Any]
        ),
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[EventTypes]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Hashable | tuple[TaskName, Hashable],
        callback: Callable,
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter], bool] | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskEvent | SchedulerEvent | tuple[TaskName, TaskEvent] | Hashable,
        callback: Callable[P, Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for an event.

        Args:
            event: The event to register the callback for.
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler instance.
        """
        pred = None if when is None else (lambda counts=self.counts: when(counts))
        self.event_manager.on(event, callback, when=pred, name=name)
        return self

    def on_start(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.STARTED` event.

        See [`SchedulerEvent.STARTED`][byop.scheduling.events.SchedulerEvent.STARTED]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.STARTED, callback, name=name, when=when)

    def on_finishing(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.FINISHING` event.

        See
        [`SchedulerEvent.FINISHING`][byop.scheduling.events.SchedulerEvent.FINISHING]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.FINISHING, callback, name=name, when=when)

    def on_finished(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.FINISHED` event.

        See [`SchedulerEvent.FINISHED`][byop.scheduling.events.SchedulerEvent.FINISHED]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.FINISHED, callback, name=name, when=when)

    def on_stop(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.STOP` event.

        See [`SchedulerEvent.STOP`][byop.scheduling.events.SchedulerEvent.STOP]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.STOP, callback, name=name, when=when)

    def on_timeout(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.TIMEOUT` event.

        See [`SchedulerEvent.TIMEOUT`][byop.scheduling.events.SchedulerEvent.TIMEOUT]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.TIMEOUT, callback, name=name, when=when)

    def on_empty(
        self,
        callback: Callable[[], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[EventTypes]], bool] | None = None,
    ) -> Self:
        """Register a callback for the `SchedulerEvent.EMPTY` event.

        See [`SchedulerEvent.EMPTY`][byop.scheduling.events.SchedulerEvent.EMPTY]
        for more details.

        Args:
            callback: The callback to register.
            name: The name of the callback. Defaults to `None`
                which means the name will be generated using the
                callback name.
            when: A function that takes the current event counter
                and returns a boolean. If the function returns
                `True` the callback will be called. Defaults to
                `None` which means the callback will always be
                called.

        Returns:
            The scheduler itself
        """
        return self.on(SchedulerEvent.EMPTY, callback, name=name, when=when)

    def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Stop the scheduler."""
        # NOTE: we allow args and kwargs to allow it to be easily
        # included in any callback.
        self._stop_event.set()

    exitcodes: ClassVar = ExitCode
    """The possible exitcodes of the scheduler.
    See [`ExitCode`][byop.scheduling.events.ExitCode]
    """

    SUBMITTED: ClassVar = TaskEvent.SUBMITTED
    """Event triggered when the task has been submitted.
    See [`TaskEvent.SUBMITTED`][byop.scheduling.events.TaskEvent.SUBMITTED]
    """

    DONE: ClassVar = TaskEvent.DONE
    """Event triggered when the task is done.
    See [`TaskEvent.DONE`][byop.scheduling.events.TaskEvent.DONE]
    """

    RETURNED: ClassVar = TaskEvent.RETURNED
    """Event triggered when the task has successfully returned a value.
    See [`TaskEvent.RETURNED`][byop.scheduling.events.TaskEvent.RETURNED]
    """

    NO_RETURN: ClassVar = TaskEvent.NO_RETURN
    """Event triggered when the task has not returned anything.
    See [`TaskEvent.NO_RETURN`][byop.scheduling.events.TaskEvent.NO_RETURN]
    """

    CANCELLED: ClassVar = TaskEvent.CANCELLED
    """Event triggered when the task has been cancelled.
    See [`TaskEvent.CANCELLED`][byop.scheduling.events.TaskEvent.CANCELLED]
    """

    UPDATE: ClassVar = TaskEvent.UPDATE
    """An event triggered when a task has sent something with `send`.
    See [`TaskEvent.UPDATE`][byop.scheduling.events.TaskEvent.UPDATE]
    """

    WAITING: ClassVar = TaskEvent.WAITING
    """An event triggered when a task is waiting for a response.
    See [`TaskEvent.WAITING`][byop.scheduling.events.TaskEvent.WAITING]
    """

    STARTED: ClassVar = SchedulerEvent.STARTED
    """The scheduler has started.
    See [`SchedulerEvent.STARTED`][byop.scheduling.events.SchedulerEvent.STARTED]
    """

    FINISHING: ClassVar = SchedulerEvent.FINISHING
    """The scheduler is finishing up.
    See [`SchedulerEvent.FINISHING`][byop.scheduling.events.SchedulerEvent.FINISHING]
    """

    FINISHED: ClassVar = SchedulerEvent.FINISHED
    """The scheduler has finished.
    See [`SchedulerEvent.FINISHED`][byop.scheduling.events.SchedulerEvent.FINISHED]
    """

    STOP: ClassVar = SchedulerEvent.STOP
    """The scheduler was stopped forcefully with `Scheduler.stop`.
    See [`SchedulerEvent.STOP`][byop.scheduling.events.SchedulerEvent.STOP]
    """

    TIMEOUT: ClassVar = SchedulerEvent.TIMEOUT
    """The scheduler has reached the timeout.
    See [`SchedulerEvent.TIMEOUT`][byop.scheduling.events.SchedulerEvent.TIMEOUT]
    """

    EMPTY: ClassVar = SchedulerEvent.EMPTY
    """The scheduler has an empty queue.
    See [`SchedulerEvent.EMPTY`][byop.scheduling.events.SchedulerEvent.EMPTY]
    """
