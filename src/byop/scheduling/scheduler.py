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
    overload,
)
from uuid import uuid4

from typing_extensions import Self

from byop.event_manager import EventManager
from byop.fluid import DelayedOp
from byop.scheduling.comm import Comm
from byop.scheduling.events import ExitCode, SchedulerStatus, TaskStatus
from byop.scheduling.task import CommTask, Task
from byop.types import CallbackName, Msg, TaskName

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class Scheduler:
    """A scheduler for submitting tasks to an Executor."""

    task: Final[type[TaskStatus]] = TaskStatus
    status: Final[type[SchedulerStatus]] = SchedulerStatus
    exitcode: Final[type[ExitCode]] = ExitCode

    def __init__(self, executor: Executor) -> None:
        """Initialize the scheduler.

        Args:
            executor: The dispatcher to use for submitting tasks.
            duration: The duration of the scheduler.
        """
        self.executor = executor

        # An event managers which handles task status and calls callbacks
        # NOTE: Typing the event manager is a little complicated, so we
        # forego it for now. However it is possible
        self.events: EventManager = EventManager(name="Scheduler")

        # Just quick access to the count of events that have occured
        self.counts = self.events.count

        # Not entirely needed but it's important to keep a reference
        # to what has been queued
        self.queue: dict[asyncio.Future, Task] = {}

        # This can be triggered either by `scheduler.stop` in a callback
        self._stop_event: asyncio.Event = asyncio.Event()

        # The currently open communcation with `dispatch_with_comm` workers
        self.communcations: dict[TaskName, asyncio.Task[None]] = {}

    def empty(self) -> bool:
        """Check if the scheduler is empty."""
        return len(self.queue) == 0

    def dispatch(
        self,
        name: TaskName | Callable[P, R],
        f: Callable[P, R] | None = None,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Task[R]:
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
            f: The function to call.
            *args: The positional arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            The future representing the task.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            logger.warning(
                "Scheduler.dispatch called outside of an asyncio loop. "
                "This is not possible."
            )
            raise e

        if f is None:
            assert callable(name)
            f = name
            name = str(uuid4())
        else:
            name = name

        sync_future = self.executor.submit(f, *args, **kwargs)
        future = asyncio.wrap_future(sync_future, loop=loop)
        future.add_done_callback(self._on_task_complete)

        task = Task(name=name, future=future, events=self.events)
        self.queue[future] = task

        # Emit the general submitted event and the task specific submitted event
        logger.debug(f"Submitted task {task}")
        self.events.emit(TaskStatus.SUBMITTED, future)
        self.events.emit((task.name, TaskStatus.SUBMITTED), future)

        return task

    def dispatch_with_comm(
        self,
        name: TaskName,
        f: Callable[Concatenate[Comm, P], R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> CommTask[R]:
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
        scheduler_end, worker_end = Comm.create(duplex=True)
        task = self.dispatch(name, f, *(worker_end, *args), **kwargs)  # type: ignore
        comm_task: CommTask[R] = CommTask.from_task(task, comm=scheduler_end)
        self.communcations[name] = asyncio.create_task(comm_task._communicate())
        return comm_task

    def _on_task_complete(self, future: asyncio.Future) -> None:
        # Remove it fom the
        task = self.queue.pop(future, None)
        if task is None:
            logger.warning(f"Task for {future} was not found in scheduler queue!")
            return

        if isinstance(task, CommTask):
            # Get the async task that was in charge of monitoring the pipes
            # and cancel it
            async_communicate_task = self.communcations.pop(task.name, None)
            if async_communicate_task is None:
                msg = f"Task to communicate with {task} was not found in scheduler!"
                logger.warning(msg)
            else:
                async_communicate_task.cancel()

            # Close the pipe if it hasn't been closed
            task.comm.close()

        # Finally let the task handle any callbacks registered
        task._finish_up()

    async def _stop_when_queue_empty(self) -> None:
        """Stop the scheduler when the queue is empty."""
        while self.queue:
            await asyncio.wait(self.queue, return_when=asyncio.ALL_COMPLETED)

        logger.debug("Queue is empty, stopping scheduler")
        return

    async def _stop_when_triggered(self) -> None:
        """Stop the scheduler when the stop event is set."""
        await self._stop_event.wait()

        logger.debug("Stop event triggered, stopping scheduler")
        return

    async def _run_scheduler(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        with self.executor:
            self.events.emit(SchedulerStatus.STARTED, self)

            # Our stopping criterion of the scheduler
            stop_criterion: list[asyncio.Task] = []

            # Monitor for `stop` being triggered
            stop_triggered = asyncio.create_task(self._stop_when_triggered())
            stop_criterion.append(stop_triggered)

            # Monitor for the queue being empty
            if end_on_empty:
                queue_empty = asyncio.create_task(self._stop_when_queue_empty())
                stop_criterion.append(queue_empty)
            else:
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
            elif queue_empty and queue_empty.done():
                stop_reason = ExitCode.EMPTY
            elif timeout is not None:
                logger.debug(f"Timeout of {timeout} reached for scheduler")
                stop_reason = ExitCode.TIMEOUT
            else:
                logger.warning("Scheduler stopped for unknown reason!")
                stop_reason = ExitCode.UNKNOWN

            # Cancel the stopping criterion
            for stopping_criteria in stop_criterion:
                stopping_criteria.cancel()

            self.events.emit(SchedulerStatus.STOPPING, self)

            logger.debug("Shutting down scheduler executor")
            if wait:
                logger.debug("Waiting for jobs to finish in executor shutdown")

            self.executor.shutdown(wait=wait)

        # We do a manual `cancel_futures` here since dask distributed
        # executor doesn't support it.
        if stop_reason == ExitCode.TIMEOUT and not wait:
            for future in self.queue:
                if not future.done():
                    future.cancel()
        else:
            logger.debug("Waiting for futures to process...")
            queue_empty = asyncio.create_task(self._stop_when_queue_empty())
            await queue_empty

        self.events.emit(SchedulerStatus.FINISHED, self)
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
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        """Async version of `run`.

        Args:
            timeout: The maximum time to run the scheduler for.
                Defaults to `None` which means no timeout.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.

        Returns:
            The reason for the scheduler ending.
        """
        return await self._run_scheduler(
            timeout=timeout,
            end_on_empty=end_on_empty,
            wait=wait,
        )

    # On any scheduler status update
    @overload
    def on(
        self,
        event: SchedulerStatus,
        *handler: Callable[[Self], None],
        when: Callable[[Self], bool] | None = None,
        name: CallbackName | None = ...,
    ) -> Self:
        ...

    # On any task submitted, finished or cancelled
    @overload
    def on(
        self,
        event: Literal[TaskStatus.SUBMITTED, TaskStatus.FINISHED, TaskStatus.CANCELLED]
        | tuple[
            TaskName,
            Literal[TaskStatus.SUBMITTED, TaskStatus.FINISHED, TaskStatus.CANCELLED],
        ],
        *handler: Callable[[Future], Any],
        when: Callable[[Future], bool] | None = None,
        name: CallbackName | None = ...,
    ) -> Self:
        ...

    # On task success
    @overload
    def on(
        self,
        event: Literal[TaskStatus.SUCCESS],
        *handler: Callable[[R], Any],
        when: Callable[[R], bool] | None = None,
        name: CallbackName | None = ...,
    ) -> Self:
        ...

    # On task error
    @overload
    def on(
        self,
        event: Literal[TaskStatus.ERROR],
        *handler: Callable[[BaseException], Any],
        when: Callable[[BaseException], bool] | None = None,
        name: CallbackName | None = ...,
    ) -> Self:
        ...

    # On a task update
    @overload
    def on(
        self,
        event: Literal[TaskStatus.UPDATE] | tuple[TaskName, Literal[TaskStatus.UPDATE]],
        *handler: Callable[[CommTask, Msg], Any],
        when: Callable[[CommTask, Msg], bool] | None = None,
        name: CallbackName | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskStatus | SchedulerStatus | tuple[TaskName, TaskStatus],
        *handler: Callable[..., Any],
        when: Callable[..., bool] | None = None,
        name: CallbackName | None = None,
    ) -> Self:
        """Register a handler for an event."""
        for h in handler:
            self.events.on(event, h, pred=when, name=name)

        return self

    def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Stop the scheduler."""
        # NOTE: we allow args and kwargs to allow it to be easily
        # included in any callback.
        self._stop_event.set()

    def count(self, event: TaskStatus | SchedulerStatus) -> DelayedOp[int, ...]:
        """The number of times an event has been emitted for callback predicates.

        Args:
            event: The event to count.

        Returns:
            A delayed operation that will return the number of times
            the event has been emitted once called.
        """

        def count(*args: Any, **kwargs: Any) -> int:  # noqa: ARG001
            return self.events.count[event]

        return DelayedOp(count)
