"""A scheduler which uses asyncio and an executor to run tasks concurrently.

It's primary use is to dispatch tasks to an executor and manage callbacks
for when they complete.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import (
    Executor,
    Future as SyncFuture,
    ProcessPoolExecutor,
    wait as wait_futures,
)
from enum import Enum, auto
from multiprocessing.context import BaseContext
from typing import Any, Callable, Hashable, TypeVar
from typing_extensions import ParamSpec, Self

from byop.events import Event, EventManager
from byop.scheduling.termination_strategies import termination_strategy

P = ParamSpec("P")
R = TypeVar("R")

CallableT = TypeVar("CallableT", bound=Callable)

logger = logging.getLogger(__name__)


class Scheduler:
    """A scheduler for submitting tasks to an Executor.

    ```python
    from byop.scheduling import Scheduler

    # For your own custom Executor
    scheduler = Scheduler(executor=...)

    # Create a scheduler which uses local processes as workers
    scheduler = Scheduler.with_processes(2)
    ```
    """

    STARTED: Event[[]] = Event("scheduler-started")
    """The scheduler has started.

    This means the scheduler has started up the executor and is ready to
    start deploying tasks to the executor.
    """

    FINISHING: Event[[]] = Event("scheduler-finishing")
    """The scheduler is finishing.

    This means the executor is still running but the stopping criterion
    for the scheduler are no longer monitored. If using `run(..., wait=True)`
    which is the deafult, the scheduler will wait until the queue as been
    emptied before reaching FINISHED.
    """

    FINISHED: Event[[]] = Event("scheduler-finished")
    """The scheduler has finished.

    This means the scheduler has stopped running the executor and
    has processed all futures and events. This is the last event
    that will be emitted from the scheduler before ceasing.
    """

    STOP: Event[[]] = Event("scheduler-stop")
    """The scheduler has been stopped explicitly.

    This means the executor is no longer running so no further tasks can be
    dispatched. The scheduler is in a state where it will wait for the current
    queue to empty out (if `run(..., wait=True)`) and for any futures to be
    processed.
    """

    TIMEOUT: Event[[]] = Event("scheduler-timeout")
    """The scheduler has reached the timeout.

    This means the scheduler reached the timeout stopping criterion, which
    is only active when `run(..., timeout: float)` was used to start the
    scheduler.
    """

    EMPTY: Event[[]] = Event("scheduler-empty")
    """The scheduler has an empty queue.

    This means the scheduler has no more running tasks in it's queue.
    This event will only trigger when `run(..., end_on_empty=False)`
    was used to start the scheduler.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        terminate: Callable[[Executor], None] | bool = True,
        event_manager: EventManager | None = None,
    ) -> None:
        """Initialize a scheduler.

        !!! note "Forcibully Terminating Workers"

            As an `Executor` does not provide an interface to forcibly
            terminate workers, we provide `terminate` as a custom
            strategy for cleaning up a provided executor. It is not possible
            to terminate running thread based workers, for example using
            `ThreadPoolExecutor` and any Executor using threads to spawn
            tasks will have to wait until all running tasks are finish
            before python can close.

            It's likely `terminate` will trigger the `EXCEPTION` event for
            any tasks that are running during the shutdown, **not***
            a cancelled event. This is because we use a
            [`Future`][concurrent.futures.Future]
            under the hood and these can not be cancelled once running.
            However there is no gaurantee of this and is up to how the
            `Executor` handles this.

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
            event_manager: An event manager to use for managing events.
                If not provided, a new one will be created.

        """
        self.executor = executor

        self.terminate: Callable[[Executor], None] | None
        if terminate is True:
            self.terminate = termination_strategy(executor)
        else:
            self.terminate = terminate if callable(terminate) else None

        # An event managers which handles task status and calls callbacks
        if event_manager is None:
            self.event_manager = EventManager(name="Scheduler-Events")
        else:
            self.event_manager = event_manager

        # The current state of things and references to them
        self.queue: list[SyncFuture] = []

        # This can be triggered either by `scheduler.stop` in a callback
        self._stop_event: asyncio.Event | None = None

        # This is a condition to make sure monitoring the queue will wait
        # properly
        self._queue_has_items: asyncio.Event | None = None

        # This is triggered when run is called
        self._running: asyncio.Event | None = None

        self.on_start = self.event_manager.subscriber(self.STARTED)
        self.on_finishing = self.event_manager.subscriber(self.FINISHING)
        self.on_finished = self.event_manager.subscriber(self.FINISHED)
        self.on_stop = self.event_manager.subscriber(self.STOP)
        self.on_timeout = self.event_manager.subscriber(self.TIMEOUT)
        self.on_empty = self.event_manager.subscriber(self.EMPTY)

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
        if self._running is None:
            return False

        return self._running.is_set()

    @property
    def counts(self) -> dict[Hashable, int]:
        """The event counter.

        Useful for predicates, for example
        ```python
        from byop.scheduling import Task

        my_scheduler.on_task_finished(
            do_something,
            when=lambda sched: sched.counts[Task.FINISHED] > 10
        )
        ```
        """
        return dict(self.event_manager.counts)

    def submit(
        self,
        function: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SyncFuture[R] | None:
        """Submits a callable to be executed with the given arguments.

        Args:
            function: The callable to be executed as
                fn(*args, **kwargs) that returns a Future instance representing
                the execution of the callable.
            args: positional arguments to pass to the function
            kwargs: keyword arguments to pass to the function

        Returns:
            A Future representing the given call.
        """
        if self._queue_has_items is None:
            raise RuntimeError("The scheduler is not running!")

        try:
            sync_future: SyncFuture = self.executor.submit(function, *args, **kwargs)
        except RuntimeError:
            logger.warning(f"Executor is not running, cannot submit task {function}")
            return None

        self.queue.append(sync_future)
        sync_future.add_done_callback(self._register_complete)
        self._queue_has_items.set()
        return sync_future

    def _register_complete(self, future: SyncFuture) -> None:
        try:
            self.queue.remove(future)
        except ValueError as e:
            logger.error(f"{future=} was not found in the queue {self.queue}: {e}!")

    async def _stop_when_queue_empty(self) -> None:
        """Stop the scheduler when the queue is empty."""
        while self.queue:
            async_futures = [asyncio.wrap_future(f) for f in self.queue]
            await asyncio.wait(async_futures, return_when=asyncio.ALL_COMPLETED)

        logger.debug("Scheduler queue is empty")

    async def _monitor_queue_empty(self) -> None:
        """Monitor for the queue being empty and trigger an event when it is."""
        if self._queue_has_items is None:
            raise RuntimeError("The scheduler is not running!")

        while True:
            while self.queue:
                async_futures = [asyncio.wrap_future(f) for f in self.queue]
                await asyncio.wait(async_futures, return_when=asyncio.ALL_COMPLETED)

            # Signal that the queue is now empty
            self._queue_has_items.clear()
            self.event_manager.emit(Scheduler.EMPTY)

            # Wait for an item to be in the queue
            await self._queue_has_items.wait()
            logger.debug("Queue has been filled again")

    async def _stop_when_triggered(self) -> bool:
        """Stop the scheduler when the stop event is set."""
        if self._stop_event is None:
            raise RuntimeError("The scheduler is not running!")

        await self._stop_event.wait()

        logger.debug("Stop event triggered, stopping scheduler")
        return True

    async def _run_scheduler(  # noqa: PLR0912, C901, PLR0915
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode:
        self.executor.__enter__()

        # Create all the private events in the current loop we're running int
        self._stop_event = asyncio.Event()
        self._queue_has_items = asyncio.Event()
        self._running = asyncio.Event()

        # Declare we are running
        self._running.set()

        self.event_manager.emit(Scheduler.STARTED)

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
        if stop_triggered.done() and self._stop_event.is_set():
            stop_reason = Scheduler.ExitCode.STOPPED
            self.event_manager.emit(Scheduler.STOP)
        elif queue_empty and queue_empty.done():
            stop_reason = Scheduler.ExitCode.EXHAUSTED
        elif timeout is not None:
            logger.debug(f"Timeout of {timeout} reached for scheduler")
            stop_reason = Scheduler.ExitCode.TIMEOUT
            self.event_manager.emit(Scheduler.TIMEOUT)
        else:
            logger.warning("Scheduler stopped for unknown reason!")
            stop_reason = Scheduler.ExitCode.UNKNOWN

        # Stop monitoring the queue to trigger an event
        if monitor_empty:
            monitor_empty.cancel()

        # Cancel the stopping criterion and await them
        for stopping_criteria in stop_criterion:
            stopping_criteria.cancel()

        all_tasks = [*stop_criterion]
        if monitor_empty is not None:
            all_tasks.append(monitor_empty)
        if queue_empty is not None:
            all_tasks.append(queue_empty)

        for task in all_tasks:
            task.cancel()

        await asyncio.wait(all_tasks, return_when=asyncio.ALL_COMPLETED)

        self.event_manager.emit(Scheduler.FINISHING)

        logger.debug(f"Shutting down scheduler executor with {wait=}")

        # The scheduler is now refusing jobs
        self._running.clear()
        self._running = None
        logger.debug("Scheduler has shutdown and declared as no longer running")

        if not wait:
            if self.terminate is not None:
                logger.debug(f"Terminating workers with {self.terminate}")
                self.terminate(self.executor)
            else:
                # Just try to cancel the tasks. Will cancel pending tasks
                # but executors like dask will even kill the job
                for future in self.queue:
                    future.cancel()
        else:
            logger.debug("Waiting for currently running tasks to finish.")

        # Here we wait, if we could terminate or cancel, then we wait for that
        # to happen, otherwise we are just waiting as anticipated.
        current_futures = self.queue[:]
        wait_futures(current_futures)
        self.executor.shutdown(wait=wait)

        self.event_manager.emit(Scheduler.FINISHED)
        logger.info(f"Scheduler finished with status {stop_reason}")

        # Remove all the private events
        self._stop_event = None
        self._queue_has_items = None

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
            timeout: The maximum time to run the scheduler for in
                seconds. Defaults to `None` which means no timeout and it
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

    def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Stop the scheduler."""
        # NOTE: we allow args and kwargs to allow it to be easily
        # included in any callback.
        if self._stop_event is None:
            raise RuntimeError("Scheduler has not been started yet")

        self._stop_event.set()

    class ExitCode(Enum):
        """The reason the scheduler ended."""

        STOPPED = auto()
        """The scheduler was stopped forcefully with `Scheduler.stop`."""

        TIMEOUT = auto()
        """The scheduler finished because of a timeout."""

        EXHAUSTED = auto()
        """The scheduler finished because it exhausted its queue."""

        UNKNOWN = auto()
        """The scheduler finished for an unknown reason."""
