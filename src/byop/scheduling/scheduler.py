"""A scheduler which uses asyncio and an executor to run tasks concurrently.

It's primary use is to dispatch tasks to an executor and manage callbacks
for when they complete.
"""
from __future__ import annotations

import asyncio
from asyncio import Future
from concurrent.futures import Executor
import logging
from typing import (
    Any,
    Callable,
    Concatenate,
    Final,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self

from byop.event_manager import EventManager
from byop.scheduling.comm import Comm
from byop.scheduling.events import ExitCode, SchedulerEvent, TaskEvent
from byop.scheduling.task import CommTask, CommTaskDescription, Task, TaskDescription
from byop.scheduling.termination_strategies import termination_strategy
from byop.types import CallbackName, Msg, TaskName

P = ParamSpec("P")
R = TypeVar("R")
_Executor = TypeVar("_Executor", bound=Executor)

logger = logging.getLogger(__name__)


class Scheduler:
    """A scheduler for submitting tasks to an Executor."""

    status: Final[type[SchedulerEvent]] = SchedulerEvent
    event: Final[type[TaskEvent]] = TaskEvent
    exitcode: Final[type[ExitCode]] = ExitCode

    def __init__(
        self,
        executor: _Executor,
        *,
        terminate: Callable[[_Executor], None] | bool = True,
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

        self.terminate: Callable[[_Executor], None] | None
        if terminate is True:
            self.terminate = termination_strategy(executor)
        else:
            self.terminate = terminate if callable(terminate) else None

        # An event managers which handles task status and calls callbacks
        # NOTE: Typing the event manager is a little complicated, so we
        # forego it for now. However it is possible
        self.event_manager: EventManager = EventManager(name="Scheduler")

        # Just quick access to the count of events that have occured
        self.counts = self.event_manager.count

        # The current state of things and references to them
        self.queue: dict[Future, Task] = {}

        # This can be triggered either by `scheduler.stop` in a callback
        self._stop_event: asyncio.Event = asyncio.Event()

        # This is a condition to make sure monitoring the queue will wait
        # properly
        self._queue_has_items: asyncio.Event = asyncio.Event()

        # This is triggered when run is called
        self._running: asyncio.Event = asyncio.Event()

        # The currently open communcation with `dispatch_with_comm` workers
        self.communcations: dict[TaskName, asyncio.Task[None]] = {}

    def empty(self) -> bool:
        """Check if the scheduler is empty."""
        return len(self.queue) == 0

    def running(self) -> bool:
        """Whether the scheduler is running and accepting tasks to dispatch."""
        return self._running.is_set()

    def task(
        self,
        name: TaskName,
        f: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> TaskDescription[R]:
        """Submit a task to the executor.

        Note:
            Dispatched tasks will only be sent to the executor upon
            `run` being called. What is recieved back is a TaskDescription
            which allows further events to be set up based on this task.

            Once `run` is called, submitted events will be fired off and
            callbacks called.

        Args:
            name: The name of the worker to run the task on.
                If no name is given, the first argument is assumed to be
                the function to call and a random uuid4 will be assigned as
                the name.
            f: The function to call.
            *args: The positional arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A TaskDescription which can be used to set up callbacks.
        """
        return TaskDescription(
            name=name,
            event_manager=self.event_manager,
            dispatch_f=self._dispatch,
            f=f,
            args=args,
            kwargs=kwargs,
        )

    def task_with_comm(
        self,
        name: TaskName,
        f: Callable[Concatenate[Comm, P], R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> CommTaskDescription[R]:
        """Submit a task to the executor.

        Note:
            Dispatch is intended to only be called by callbacks and
            once started with `start()`.

            If called directly it will just submit the task to the
            executor and return the future, calling no further
            registered callbacks.

        Args:
            name: The name of the worker to run the task on.
                If no name is given, the first argument is assumed to be
                the function to call and a random uuid4 will be assigned as
                the name.
            f: The function to call. Must accept a `Comm` as its first
                argument.
            *args: The positional arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            The future representing the task.
        """
        return CommTaskDescription(
            name=name,
            event_manager=self.event_manager,
            dispatch_f=self._dispatch,
            f=f,
            args=args,
            kwargs=kwargs,
        )

    @overload
    def _dispatch(self, task_desc: CommTaskDescription[R]) -> CommTask[R]:
        ...

    @overload
    def _dispatch(self, task_desc: TaskDescription[R]) -> Task[R]:
        ...

    def _dispatch(
        self,
        task_desc: CommTaskDescription[R] | TaskDescription[R],
    ) -> Task[R] | CommTask[R]:
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

        f = task_desc.f
        args = task_desc.args
        kwargs = task_desc.kwargs

        if isinstance(task_desc, CommTaskDescription):
            scheduler_comm, worker_comm = Comm.create(duplex=True)
            sync_future = self.executor.submit(f, *(worker_comm, *args), **kwargs)
        else:
            scheduler_comm = None
            sync_future = self.executor.submit(f, *args, **kwargs)  # type: ignore

        future = asyncio.wrap_future(sync_future, loop=loop)
        future.add_done_callback(self._on_task_complete)

        task = Task(future=future, desc=task_desc)

        if isinstance(task_desc, CommTaskDescription):
            assert scheduler_comm is not None
            comm_task = CommTask.from_task(task, comm=scheduler_comm)
            self.communcations[comm_task.name] = asyncio.create_task(
                self._communicate(comm_task)
            )

        # Place our future and task in the queue
        self.queue[future] = task
        self._queue_has_items.set()

        # Emit the general submitted event and the task specific submitted event
        logger.debug(f"Submitted task {task}")
        self.event_manager.emit(TaskEvent.SUBMITTED, task)
        self.event_manager.emit((task.name, TaskEvent.SUBMITTED), task)
        return task

    def _on_task_complete(self, future: asyncio.Future) -> None:
        # Remove it fom the
        task = self.queue.pop(future, None)
        if task is None:
            logger.warning(f"Task for {future} was not found in scheduler queue!")
            return

        # NOTE: I have no reason to choose whether to emit the general
        # or task specific event first. If there is a compelling argument
        # to choose one over the other, please raise an issue dicussing it!
        # @eddiebergman
        if future.cancelled():
            logger.debug(f"Task {task} was cancelled")

            self.event_manager.emit(TaskEvent.CANCELLED, task)
            self.event_manager.emit((task.name, TaskEvent.CANCELLED), task)
            return

        exception = future.exception()
        result = future.result() if exception is None else None

        logger.debug(f"Task {task} finished")
        self.event_manager.emit(TaskEvent.FINISHED, task)
        self.event_manager.emit((task.name, TaskEvent.FINISHED), task)

        if exception is None:
            logger.debug(f"Task {task} completed successfully")
            self.event_manager.emit(TaskEvent.SUCCESS, result)
            self.event_manager.emit((task.name, TaskEvent.SUCCESS), result)
        else:
            logger.debug(f"Task {task} failed with {exception}")
            self.event_manager.emit(TaskEvent.ERROR, exception)
            self.event_manager.emit((task.name, TaskEvent.ERROR), exception)

        # If dealing with a comm task, get the async task that was in
        # charge of monitoring the pipes and cancel it, as well as close
        # remaining comms
        if isinstance(task, CommTask):
            async_communicate_task = self.communcations.pop(task.name, None)
            if async_communicate_task is None:
                msg = f"Task to communicate with {task} was not found in scheduler!"
                logger.warning(msg)
            else:
                async_communicate_task.cancel()

            # Close the pipe if it hasn't been closed
            task.comm.close()

    async def _communicate(self, task: CommTask) -> None:
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
            self.event_manager.emit(SchedulerEvent.EMPTY, self)

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

        self.event_manager.emit(SchedulerEvent.STARTED, self)

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
            self.event_manager.emit(SchedulerEvent.STOPPED, self)
        elif queue_empty and queue_empty.done():
            stop_reason = ExitCode.EMPTY
        elif timeout is not None:
            logger.debug(f"Timeout of {timeout} reached for scheduler")
            stop_reason = ExitCode.TIMEOUT
            self.event_manager.emit(SchedulerEvent.TIMEOUT, self)
        else:
            logger.warning("Scheduler stopped for unknown reason!")
            stop_reason = ExitCode.UNKNOWN

        # Stop monitoring the queue to trigger an event
        if monitor_empty:
            monitor_empty.cancel()

        # Cancel the stopping criterion
        for stopping_criteria in stop_criterion:
            stopping_criteria.cancel()

        self.event_manager.emit(SchedulerEvent.STOPPING, self)

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

        self.event_manager.emit(SchedulerEvent.FINISHED, self)
        logger.info(f"Scheduler finished with status {stop_reason}")
        return stop_reason

    def run(
        self,
        initial: TaskDescription | list[TaskDescription] | None = None,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        """Run the scheduler.

        Args:
            initial: The initial tasks to run. Defaults to `None`
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

        if initial:
            if isinstance(initial, TaskDescription):
                initial = [initial]

            for task in initial:
                self.on(SchedulerEvent.STARTED, task, name=task.name)

        return loop.run_until_complete(
            self._run_scheduler(timeout=timeout, end_on_empty=end_on_empty, wait=wait)
        )

    async def async_run(
        self,
        initial: TaskDescription | list[TaskDescription] | None = None,
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
            if isinstance(initial, TaskDescription):
                initial = [initial]

            for task in initial:
                self.on(SchedulerEvent.STARTED, task, name=task.name)

        return await self._run_scheduler(
            timeout=timeout,
            end_on_empty=end_on_empty,
            wait=wait,
        )

    # On any scheduler status update
    @overload
    def on(
        self,
        event: SchedulerEvent,
        handler: Callable[[Self], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Self], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    # On any task submitted, finished or cancelled
    @overload
    def on(
        self,
        event: Literal[TaskEvent.SUBMITTED, TaskEvent.FINISHED, TaskEvent.CANCELLED]
        | tuple[
            TaskName,
            Literal[TaskEvent.SUBMITTED, TaskEvent.FINISHED, TaskEvent.CANCELLED],
        ],
        handler: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Task[R]], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    # On task success
    @overload
    def on(
        self,
        event: Literal[TaskEvent.SUCCESS],
        handler: Callable[[R], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[R], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    # On task error
    @overload
    def on(
        self,
        event: Literal[TaskEvent.ERROR],
        handler: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[BaseException], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    # On a task update
    @overload
    def on(
        self,
        event: Literal[TaskEvent.UPDATE] | tuple[TaskName, Literal[TaskEvent.UPDATE]],
        handler: Callable[[CommTask, Msg], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[CommTask, Msg], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    # On a task waiting
    @overload
    def on(
        self,
        event: Literal[TaskEvent.WAITING] | tuple[TaskName, Literal[TaskEvent.WAITING]],
        handler: Callable[[CommTask], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[CommTask], bool] | None = ...,
        every: int | None = ...,
        count: Callable[[int], bool] | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskEvent | SchedulerEvent | tuple[TaskName, TaskEvent],
        handler: Callable[P, Any],
        *,
        when: Callable[P, bool] | None = None,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
    ) -> Self:
        """Register a handler for an event."""
        self.event_manager.on(
            event, handler, when=when, name=name, every=every, count=count
        )
        return self

    def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Stop the scheduler."""
        # NOTE: we allow args and kwargs to allow it to be easily
        # included in any callback.
        self._stop_event.set()

    def on_start(
        self,
        handler: Callable[[Self], Any],
        *,
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for when the scheduler.

        See SchedulerStatus.STARTED.

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        return self.on(SchedulerEvent.STARTED, handler, when=when, name=name)

    def on_timeout(
        self,
        handler: Callable[[Self], Any],
        *,
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for when the scheduler times out.

        See SchedulerStatus.TIMEOUT.

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        return self.on(SchedulerEvent.TIMEOUT, handler, when=when, name=name)

    def on_stopping(
        self,
        handler: Callable[[Self], Any],
        *,
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for when the scheduler is stopping.

        See SchedulerStatus.STOPPING.

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        return self.on(SchedulerEvent.STOPPING, handler, when=when, name=name)

    def on_stopped(
        self,
        handler: Callable[[Self], Any],
        *,
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for when the scheduler has STOPPED.

        Seed SchedulerStatus.STOPPED

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        return self.on(SchedulerEvent.STOPPED, handler, when=when, name=name)

    def on_finished(
        self,
        handler: Callable[[Self], Any],
        *,
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for when the scheduler has FINISHED.

        See SchedulerStatus.FINISHED

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        return self.on(SchedulerEvent.FINISHED, handler, when=when, name=name)

    def on_empty(
        self,
        handler: Callable[[Self], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self], bool] | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
    ) -> Self:
        """Register a handler for when the scheduler has an EMPTY queue.

        See SchedulerStatus.EMPTY.

        Args:
            handler: The handler to call.
            when: A predicate to check before calling the handler. Defaults to `None`
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            The scheduler.
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
        event = SchedulerEvent.EMPTY
        return self.on(event, handler, when=when, every=every, count=count, name=name)

    def on_task_submit(
        self,
        handler: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when a task is submitted to the scheduler.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[Task[R]], Any], handler)

        event = TaskEvent.SUBMITTED
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_finish(
        self,
        handler: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when a task is finished.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[Task[R]], Any], handler)

        event = TaskEvent.FINISHED
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_success(
        self,
        handler: Callable[[R], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[R], bool] | None = None,
    ) -> Self:
        """Called when a task is finished successfuly and has a result.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[R], Any], handler)

        event = TaskEvent.SUCCESS
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_error(
        self,
        handler: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[BaseException], bool] | None = None,
    ) -> Self:
        """Called when a task is finished but raised an unacught error during execution.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[BaseException], Any], handler)

        event = TaskEvent.ERROR
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_cancelled(
        self,
        handler: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when a task cancelled and will not finish.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[Task[R]], Any], handler)

        event = TaskEvent.CANCELLED
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_update(
        self,
        handler: Callable[[CommTask[R], Msg], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[CommTask[R], Msg], bool] | None = None,
    ) -> Self:
        """Called when a task sends an update with `Comm.send`.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[CommTask[R], Msg], Any], handler)

        event = TaskEvent.UPDATE
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    def on_task_waiting(
        self,
        handler: Callable[[CommTask[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[CommTask[R]], bool] | None = None,
    ) -> Self:
        """Called when a task sends is waiting for `Comm.recv`.

        Args:
            handler: The function to call
            name: The name of the handler. Defaults to `None`
                If a `Task` is provided, it will use the `task.name` if no `name` is
                provided.
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(handler, TaskDescription):
            name = name if name else handler.name
            handler = cast(Callable[[CommTask[R]], Any], handler)

        event = TaskEvent.WAITING
        self.event_manager.on(
            event,
            handler,
            name=name,
            every=every,
            count=count,
            when=when,  # type: ignore
        )
        return self

    @property
    def event_counts(self) -> dict[TaskEvent, int]:
        """The event counter.

        Useful for predicates, for example
        ```python
        from byop.scheduling import TaskEvent

        my_scheduler.on_task_finished(
            do_something,
            when=lambda sched: sched.event_counts[TaskEvent.FINISHED] > 10
        )
        ```
        """
        return {event: self.event_manager.count[event] for event in TaskEvent}

    @property
    def status_counts(self) -> dict[SchedulerEvent, int]:
        """The event counter.

        Useful for predicates, for example
        ```python
        from byop.scheduling import SchedulerEvent

        my_scheduler.on_task_finished(
            do_something,
            when=lambda sched: sched.event_counts[TaskEvent.FINISHED] > 10
        )
        ```
        """
        return {event: self.event_manager.count[event] for event in SchedulerEvent}
