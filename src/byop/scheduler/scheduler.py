"""A scheduler which uses asyncio and an executor to run tasks concurrently.

It's primary use is to dispatch tasks to an executor and manage callbacks
for when they complete.
"""
from __future__ import annotations

import asyncio
from asyncio import Future, Task
from concurrent.futures import Executor
from contextlib import contextmanager
from functools import partial
from itertools import chain
import logging
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)

from typing_extensions import Self

from byop.event_manager import EventManager
from byop.fluid import ChainablePredicate, DelayedOp, Partial
from byop.scheduler.events import ExitCode, SchedulerStatus, Signal, TaskStatus

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def _signal_raised(
    check_for: Signal,
    *signal_sources: Iterable[Signal] | None,
) -> bool:
    """Check if any callback raised a a given signal."""
    sources = [source for source in signal_sources if source is not None]
    return any(s is check_for for s in chain.from_iterable(sources))


class Scheduler:
    """A scheduler for submitting tasks to an Executor."""

    task: Final[type[TaskStatus]] = TaskStatus
    status: Final[type[SchedulerStatus]] = SchedulerStatus
    signal: Final[type[Signal]] = Signal
    exitcode: Final[type[ExitCode]] = ExitCode

    def __init__(self, executor: Executor) -> None:
        """Initialize the scheduler.

        Args:
            executor: The dispatcher to use for submitting tasks.
            duration: The duration of the scheduler.
        """
        self.executor = executor

        # An event managers which handles task status and scheduler status
        # events with any callback that can return either a `SchedulerSignal`
        # or `None`.
        self.events: EventManager[TaskStatus | SchedulerStatus, Signal]
        self.events = EventManager(name="Scheduler")

        # Just quick access to the count of events that have occured
        self.counts = self.events.count

        # Not entirely needed but it's important to keep a reference
        # to what has been queued
        self.queue: list[asyncio.Future] = []

        # This can be triggered either by a timeout or with a call to stop
        # or with a callback that returns a `SchedulerSignal.STOP`
        self._stop_event: asyncio.Event = asyncio.Event()

    def empty(self) -> bool:
        """Check if the scheduler is empty."""
        return len(self.queue) == 0

    def dispatch(
        self,
        f: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        """Submit a task to the executor.

        Note:
            Dispatch is intended to only be called by callbacks and
            once started with `start()`.

            If called directly it will just submit the task to the
            executor and return the future, calling no further
            registered callbacks.

        Args:
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

        future = loop.run_in_executor(self.executor, f, *args, **kwargs)
        logger.debug(f"Submitted task ({f}) {future}")

        future.add_done_callback(self._process_future)
        self.queue.append(future)
        self.events.emit(TaskStatus.SUBMITTED, future)
        return future

    def _process_future(self, future: Future) -> None:
        logger.debug(f"Processing future {future}")
        if future in self.queue:
            self.queue.remove(future)

        if future.cancelled():
            logger.debug(f"Future {future} was cancelled")
            signals = self.events.emit(TaskStatus.CANCELLED, future)
        else:
            exception = future.exception()
            result = future.result() if exception is None else None

            logger.debug(f"Future {future} finished")
            self.events.emit(TaskStatus.FINISHED, result, exception)

            if exception is None:
                logger.debug(f"Future {future} completed successfully")
                signals = self.events.emit(TaskStatus.COMPLETE, result)
            else:
                logger.debug(f"Future {future} failed with {exception}")
                signals = self.events.emit(TaskStatus.ERROR, exception)

        if _signal_raised(Signal.STOP, signals):
            logger.debug("Received stop signal")
            self.stop()

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
            stop_criterion: list[Task] = []

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

    @overload
    def on(
        self,
        event: SchedulerStatus,
        *handler: Callable[[Self], None],
        when: Callable[[Self], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskStatus.SUBMITTED],
        *handler: Callable[[Future], None],
        when: Callable[[Future], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskStatus.CANCELLED],
        *handler: Callable[[Future], Signal | None],
        when: Callable[[Future], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskStatus.FINISHED],
        *handler: Callable[[Any, Exception], Signal | None],
        when: Callable[[Any, Exception], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskStatus.COMPLETE],
        *handler: Callable[[Any], Signal | None],
        when: Callable[[Any], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskStatus.ERROR],
        *handler: Callable[[Exception], Signal | None],
        when: Callable[[Exception], bool] | None = None,
        name: str | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskStatus | SchedulerStatus,
        *handler: Callable[P, Signal | None],
        when: Callable[P, bool] | None = None,
        name: str | None = None,
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

    @contextmanager
    def when(
        self,
        event: TaskStatus | SchedulerStatus,
        *preds: Callable[P, bool],
    ) -> Iterator[Partial]:
        """A context manager to register a callback for an event.

        Args:
            event: The event to register the callback for.
            *preds: The predicates to use which determine if
                the callback should be called.

        Returns:
            A partial function that can be called with the callback
        """
        if len(preds) == 0:
            yield partial(self.events.on, event)
        else:
            yield partial(self.on, event=event, when=ChainablePredicate.all(*preds))

    def on_task_finish(
        self, *handler: Callable[[Any, Exception], None], name: str | None = None
    ) -> Self:
        """Register handler(s) for a task finishing.

        Args:
            handler: The handler(s) to register.
            name: The name of the handler(s). Defaults to `None`.

        Returns:
            The scheduler instance.
        """
        return self.on(TaskStatus.FINISHED, *handler, name=name)
