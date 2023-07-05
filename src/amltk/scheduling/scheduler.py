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
from threading import Timer
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from amltk.asyncm import ContextEvent
from amltk.events import Emitter, Event
from amltk.functional import Flag
from amltk.scheduling.sequential_executor import SequentialExecutor
from amltk.scheduling.termination_strategies import termination_strategy

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from typing_extensions import ParamSpec, Self

    from amltk.dask_jobqueue import DJQ_NAMES

    P = ParamSpec("P")
    R = TypeVar("R")

    CallableT = TypeVar("CallableT", bound=Callable)

logger = logging.getLogger(__name__)


class Scheduler(Emitter):
    """A scheduler for submitting tasks to an Executor.

    ```python
    from amltk.scheduling import Scheduler

    # For your own custom Executor
    scheduler = Scheduler(executor=...)

    # Create a scheduler which uses local processes as workers
    scheduler = Scheduler.with_processes(2)

    # Run a function when the scheduler starts, twice
    @scheduler.on_start(repeat=2)
    def say_hello_world():
        print("hello world")

    @scheduler.on_finish
    def say_goodbye_world():
        print("goodbye world")

    scheduler.run(timeout=10)
    ```

    Attributes:
        executor: The executor to use to run tasks.
        queue: The queue of tasks running
        on_submission: A [`Subscriber`][amltk.events.Subscriber] which is called when
            a future is submitted.
            ```python
            @task.on_submission
            def _(future: Future[R]):
                ...
            ```
        on_finishing: A [`Subscriber`][amltk.events.Subscriber] which is called when the
            scheduler is finishing up.
            ```python
            @task.on_finishing
            def _():
                ...
            ```
        on_start: A [`Subscriber`][amltk.events.Subscriber] which is called when the
            scheduler starts.

            ```python
            @task.on_start
            def _():
                ...
            ```
        on_future_done: A [`Subscriber`][amltk.events.Subscriber] which is called when
            a future is done.
            ```python
            @task.on_future_done
            def _(future: Future[R]):
                ...
            ```
        on_future_cancelled: A [`Subscriber`][amltk.events.Subscriber] which is called
            when a future is cancelled.
            ```python
            @task.on_future_cancelled
            def _(future: Future[R]):
                ...
            ```
        on_finished: A [`Subscriber`][amltk.events.Subscriber] which is called when
            the scheduler finishes.
            ```python
            @task.on_finished
            def _():
                ...
            ```
        on_stop: A [`Subscriber`][amltk.events.Subscriber] which is called when the
            scheduler is stopped.
            ```python
            @task.on_stop
            def _():
                ...
            ```
        on_timeout: A [`Subscriber`][amltk.events.Subscriber] which is called when
            the scheduler reaches the timeout.
            ```python
            @task.on_timeout
            def _():
                ...
            ```
        on_empty: A [`Subscriber`][amltk.events.Subscriber] which is called when the
            queue is empty.
            ```python
            @task.on_empty
            def _():
                ...
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

    FUTURE_SUBMITTED: Event[SyncFuture] = Event("scheduler-future-submitted")
    """The scheduler has had a future submitted to it."""

    FUTURE_DONE: Event[SyncFuture] = Event("scheduler-future-done")
    """A future submitted by the scheduler is done."""

    FUTURE_CANCELLED: Event[SyncFuture] = Event("scheduler-future-cancelled")
    """A future submitted by the scheduler was cancelled."""

    def __init__(
        self,
        executor: Executor,
        *,
        terminate: Callable[[Executor], None] | bool = True,
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
        """
        super().__init__(event_manager="Scheduler")
        self.executor = executor

        # The current state of things and references to them
        self.queue: list[SyncFuture] = []

        self.on_start = self.subscriber(self.STARTED)
        self.on_finishing = self.subscriber(self.FINISHING)
        self.on_submission = self.subscriber(self.FUTURE_SUBMITTED)
        self.on_future_done = self.subscriber(self.FUTURE_DONE)
        self.on_future_cancelled = self.subscriber(self.FUTURE_CANCELLED)
        self.on_finished = self.subscriber(self.FINISHED)
        self.on_stop = self.subscriber(self.STOP)
        self.on_timeout = self.subscriber(self.TIMEOUT)
        self.on_empty = self.subscriber(self.EMPTY)
        self._terminate: Callable[[Executor], None] | None
        if terminate is True:
            self._terminate = termination_strategy(executor)
        else:
            self._terminate = terminate if callable(terminate) else None

        # This can be triggered either by `scheduler.stop` in a callback
        self._stop_event = ContextEvent()

        # This is a condition to make sure monitoring the queue will wait properly
        self._queue_has_items_event = asyncio.Event()

        # This is triggered when run is called
        self._running_event = asyncio.Event()

        # This is set once `run` is called
        self._end_on_exception_flag = Flag(initial=False)

        # This is used to manage suequential queues, where we need a Thread
        # timer to ensure that we don't get caught in an endless loop waiting
        # for the `timeout` in `_run_scheduler` to trigger. This won't trigger
        # because the sync code of submit could possibly keep calling itself
        # endlessly, preventing any of the async code from running.
        self._timeout_timer: Timer | None = None

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

    @classmethod
    def with_sequential(cls) -> Self:
        """Create a Scheduler that runs sequentially.

        This is useful for debugging and testing. Uses
        a [`SequentialExecutor`][amltk.scheduling.SequentialExecutor].
        """
        return cls(executor=SequentialExecutor())

    @classmethod
    def with_slurm(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a SLURM cluster.

        This is useful for running on a SLURM cluster. Uses
        [dask_jobqueue.SLURMCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a SLURM cluster.
        """
        return cls.with_dask_jobqueue(
            "slurm",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_pbs(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a PBS cluster.

        This is useful for running on a PBS cluster. Uses
        [dask_jobqueue.PBSCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a PBS cluster.
        """
        return cls.with_dask_jobqueue(
            "pbs",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_sge(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a SGE cluster.

        This is useful for running on a SGE cluster. Uses
        [dask_jobqueue.SGECluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a SGE cluster.
        """
        return cls.with_dask_jobqueue(
            "sge",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_oar(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a OAR cluster.

        This is useful for running on a OAR cluster. Uses
        [dask_jobqueue.OARCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a OAR cluster.
        """
        return cls.with_dask_jobqueue(
            "oar",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_moab(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a Moab cluster.

        This is useful for running on a Moab cluster. Uses
        [dask_jobqueue.MoabCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a Moab cluster.
        """
        return cls.with_dask_jobqueue(
            "moab",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_lsf(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a LSF cluster.

        This is useful for running on a LSF cluster. Uses
        [dask_jobqueue.LSFCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a LSF cluster.
        """
        return cls.with_dask_jobqueue(
            "lsf",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_htcondor(
        cls,
        *,
        n_workers: int,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a HTCondor cluster.

        This is useful for running on a HTCondor cluster. Uses
        [dask_jobqueue.HTCondorCluster][].

        Args:
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Returns:
            A scheduler that will run on a HTCondor cluster.
        """
        return cls.with_dask_jobqueue(
            "htcondor",
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_dask_jobqueue(
        cls,
        name: DJQ_NAMES,
        *,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        n_workers: int,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler with using `dask-jobqueue`.

        See [`dask_jobqueue`][dask_jobqueue] for more details.

        [dask_jobqueue]: https://jobqueue.dask.org/en/latest/

        Args:
            name: The name of the jobqueue to use. This is the name of the
                class in `dask_jobqueue` to use. For example, to use
                `dask_jobqueue.SLURMCluster`, you would use `slurm`.
            n_workers: The number of workers to start.
            submit_command: Overwrite the command to submit a worker if necessary.
            cancel_command: Overwrite the command to cancel a worker if necessary.
            kwargs: Any additional keyword arguments to pass to the
                `dask_jobqueue` class.

        Raises:
            ImportError: If `dask-jobqueue` is not installed.

        Returns:
            A new scheduler with a `dask_jobqueue` executor.
        """
        try:
            from amltk.dask_jobqueue import DaskJobqueueExecutor

        except ImportError as e:
            raise ImportError(
                f"To use the {name} executor, you must install the "
                "`dask-jobqueue` package.",
            ) from e

        executor = DaskJobqueueExecutor.from_str(
            name,
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )
        return cls(executor)

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
        return self._running_event.is_set()

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
        if not self.running():
            logger.info(f"Scheduler is not running, cannot submit task {function}")
            return None

        try:
            sync_future: SyncFuture = self.executor.submit(function, *args, **kwargs)
        except RuntimeError:
            logger.warning(f"Executor is not running, cannot submit task {function}")
            return None

        self._register_future(sync_future)
        return sync_future

    def _register_future(self, future: SyncFuture) -> None:
        self.queue.append(future)
        self._queue_has_items_event.set()

        self.emit(self.FUTURE_SUBMITTED, future)
        future.add_done_callback(self._register_complete)

    def _register_complete(self, future: SyncFuture) -> None:
        try:
            self.queue.remove(future)

        except ValueError as e:
            logger.error(f"{future=} was not found in the queue {self.queue}: {e}!")

        if future.cancelled():
            self.emit(self.FUTURE_CANCELLED, future)
            return

        self.emit(self.FUTURE_DONE, future)

        exception = future.exception()
        if self._end_on_exception_flag and future.done() and exception:
            self.stop("Ending on first exception", exception=exception)

    async def _stop_when_queue_empty(self) -> None:
        """Stop the scheduler when the queue is empty."""
        while self.queue:
            async_futures = [asyncio.wrap_future(f) for f in self.queue]
            await asyncio.wait(async_futures, return_when=asyncio.ALL_COMPLETED)

        logger.debug("Scheduler queue is empty")

    async def _monitor_queue_empty(self) -> None:
        """Monitor for the queue being empty and trigger an event when it is."""
        if not self.running():
            raise RuntimeError("The scheduler is not running!")

        while True:
            while self.queue:
                async_futures = [asyncio.wrap_future(f) for f in self.queue]
                await asyncio.wait(async_futures, return_when=asyncio.ALL_COMPLETED)

            # Signal that the queue is now empty
            self._queue_has_items_event.clear()
            self.emit(Scheduler.EMPTY)

            # Wait for an item to be in the queue
            await self._queue_has_items_event.wait()
            logger.debug("Queue has been filled again")

    async def _stop_when_triggered(self) -> bool:
        """Stop the scheduler when the stop event is set."""
        if not self.running():
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
    ) -> ExitCode | BaseException:
        self.executor.__enter__()

        # Declare we are running
        self._running_event.set()

        # Start a Thread Timer as our timing mechanism.
        # HACK: This is required because the SequentialExecutor mode
        # will not allow the async loop to run, meaning we can't update
        # any internal state.
        if timeout is not None:
            self._timeout_timer = Timer(timeout, lambda: None)
            self._timeout_timer.start()

        self.emit(Scheduler.STARTED)

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
        stop_reason: BaseException | Scheduler.ExitCode
        if stop_triggered.done() and self._stop_event.is_set():
            stop_reason = Scheduler.ExitCode.STOPPED

            msg, exception = self._stop_event.context
            if msg and exception:
                msg = "\n".join([msg, f"{type(exception)}: {exception}"])
            elif msg and exception:
                msg = f"Scheduler stopped with message:\n{msg}"
            elif exception:
                msg = "\n".join(
                    [
                        f"Scheduler stopped with exception {type(exception)}:",
                        f"{exception}",
                    ],
                )
            else:
                msg = "Scheduler had `stop()` called on it."

            if msg:
                logger.debug(msg)

            self.emit(Scheduler.STOP)
            if self._end_on_exception_flag and exception:
                stop_reason = exception
            else:
                stop_reason = Scheduler.ExitCode.STOPPED

        elif queue_empty and queue_empty.done():
            logger.debug("Scheduler stopped due to being empty.")
            stop_reason = Scheduler.ExitCode.EXHAUSTED
        elif timeout is not None:
            logger.debug(f"Scheduler stopping as {timeout=} reached.")
            stop_reason = Scheduler.ExitCode.TIMEOUT
            self.emit(Scheduler.TIMEOUT)
        else:
            logger.warning("Scheduler stopping for unknown reason!")
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

        self.emit(Scheduler.FINISHING)
        logging.info("Scheduler is finished")

        logger.debug(f"Shutting down scheduler executor with {wait=}")

        # The scheduler is now refusing jobs
        self._running_event.clear()
        logger.debug("Scheduler has shutdown and declared as no longer running")

        if not wait:
            if self._terminate is not None:
                logger.debug(f"Terminating workers with {self._terminate }")
                self._terminate(self.executor)
            else:
                # Just try to cancel the tasks. Will cancel pending tasks
                # but executors like dask will even kill the job
                for future in self.queue:
                    if not future.done():
                        logger.debug(f"Cancelling {future=}")
                        future.cancel()
        else:
            logger.debug("Waiting for currently running tasks to finish.")

        # Here we wait, if we could terminate or cancel, then we wait for that
        # to happen, otherwise we are just waiting as anticipated.
        current_futures = self.queue[:]
        wait_futures(current_futures)
        self.executor.shutdown(wait=wait)

        self.emit(Scheduler.FINISHED)
        logger.info(f"Scheduler finished with status {stop_reason}")

        # Clear all events
        self._stop_event.clear()
        self._queue_has_items_event.clear()

        return stop_reason

    def run(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
        end_on_exception: bool = True,
        raises: bool = True,
        asyncio_debug_mode: bool = False,
    ) -> ExitCode | BaseException:
        """Run the scheduler.

        Args:
            timeout: The maximum time to run the scheduler for in
                seconds. Defaults to `None` which means no timeout and it
                will end once the queue becomes empty.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.
            end_on_exception: Whether to end if an exception occurs.
            raises: Whether to raise an exception if the scheduler
                ends due to an exception. Has no effect if `end_on_exception`
                is `False`.
            asyncio_debug_mode: Whether to run the async loop in debug mode.
                Defaults to `False`. Please see [`asyncio.run`][] for more.

        Returns:
            The reason for the scheduler ending.

        Raises:
            RuntimeError: If the scheduler is already running.
        """
        return asyncio.run(
            self.async_run(
                timeout=timeout,
                end_on_empty=end_on_empty,
                wait=wait,
                end_on_exception=end_on_exception,
                raises=raises,
            ),
            debug=asyncio_debug_mode,
        )

    async def async_run(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
        end_on_exception: bool = True,
        raises: bool = True,
    ) -> ExitCode | BaseException:
        """Async version of `run`.

        Args:
            timeout: The maximum time to run the scheduler for.
                Defaults to `None` which means no timeout.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.
            end_on_exception: Whether to end if an exception occurs.
            raises: Whether to raise an exception if the scheduler
                ends due to an exception. Has no effect if `end_on_exception`
                is `False`.

        Returns:
            The reason for the scheduler ending.
        """
        if self.running():
            raise RuntimeError("Scheduler already seems to be running")

        logger.debug("Starting scheduler")

        # Make sure the flag is set
        self._end_on_exception_flag.set(value=end_on_exception)

        result = await self._run_scheduler(
            timeout=timeout,
            end_on_empty=end_on_empty,
            wait=wait,
        )

        # Reset it back to its default
        self._end_on_exception_flag.reset()

        # If we were meant to end on an exception and the result
        # we got back from the scheduler was an exception, raise it
        if isinstance(result, BaseException):
            if raises:
                raise result

            return result

        return result

    def stop(self, *args: Any, **kwargs: Any) -> None:
        """Stop the scheduler.

        The scheduler will stop, finishing currently running tasks depending
        on the `wait=` parameter to [`Scheduler.run`][amltk.Scheduler.run].

        The call signature is kept open with `*args, **kwargs` to make it
        easier to include in any callback.

        Args:
            *args: Logged in a debug message
            **kwargs: Logged in a debug message
                **stop_msg**: The message to pass to the stop event which
                    gets logged as the stop reason.
                **exception**: The exception to pass to the stop event which
                gets logged as the stop reason.
        """
        if not self.running():
            return

        msg = kwargs.get("stop_msg", "stop() called")

        self._stop_event.set(msg=msg, exception=kwargs.get("exception"))
        logger.debug(f"Stop event set with {args=} and {kwargs=}")
        self._running_event.clear()

    class ExitCode(Enum):
        """The reason the scheduler ended."""

        STOPPED = auto()
        """The scheduler was stopped forcefully with `Scheduler.stop`."""

        TIMEOUT = auto()
        """The scheduler finished because of a timeout."""

        EXHAUSTED = auto()
        """The scheduler finished because it exhausted its queue."""

        CANCELLED = auto()
        """The scheduler was cancelled."""

        UNKNOWN = auto()
        """The scheduler finished for an unknown reason."""
