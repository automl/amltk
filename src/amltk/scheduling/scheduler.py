"""A scheduler which uses asyncio and an executor to run tasks concurrently.

It's primary use is to dispatch tasks to an executor and manage callbacks
for when they complete.
"""
from __future__ import annotations

import asyncio
import logging
import warnings
from asyncio import Future
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from threading import Timer
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeVar
from uuid import uuid4

from amltk.asyncm import ContextEvent
from amltk.events import Emitter, Event, Subscriber
from amltk.exceptions import SchedulerNotRunningError
from amltk.functional import Flag
from amltk.scheduling.sequential_executor import SequentialExecutor
from amltk.scheduling.task import Task
from amltk.scheduling.termination_strategies import termination_strategy

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from typing_extensions import ParamSpec, Self

    from rich.console import RenderableType
    from rich.live import Live

    from amltk.dask_jobqueue import DJQ_NAMES
    from amltk.scheduling.task_plugin import TaskPlugin

    P = ParamSpec("P")
    R = TypeVar("R")

    CallableT = TypeVar("CallableT", bound=Callable)

logger = logging.getLogger(__name__)


@dataclass
class ExitState:
    """The exit state of a scheduler.

    Attributes:
        reason: The reason for the exit.
        exception: The exception that caused the exit, if any.
    """

    code: Scheduler.ExitCode
    exception: BaseException | None = None


class Scheduler:
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
    """

    executor: Executor
    """The executor to use to run tasks."""

    emitter: Emitter
    """The emitter to use for events."""

    queue: dict[Future, tuple[Callable, tuple, dict]]
    """The queue of tasks running."""

    on_start: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when the
    scheduler starts.

    ```python
    @scheduler.on_start
    def my_callback():
        ...
    ```
    """
    on_future_submitted: Subscriber[Future]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    a future is submitted.

    ```python
    @scheduler.on_submission
    def my_callback(future: Future):
        ...
    ```
    """
    on_future_done: Subscriber[Future]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    a future is done.

    ```python
    @scheduler.on_future_done
    def my_callback(future: Future):
        ...
    ```
    """
    on_future_result: Subscriber[Future, Any]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    a future returned with a result.

    ```python
    @scheduler.on_future_result
    def my_callback(future: Future, result: Any):
        ...
    ```
    """
    on_future_exception: Subscriber[Future, BaseException]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    a future has an exception.

    ```python
    @scheduler.on_future_exception
    def my_callback(future: Future, exception: BaseException):
        ...
    ```
    """
    on_future_cancelled: Subscriber[Future]
    """A [`Subscriber`][amltk.events.Subscriber] which is called
    when a future is cancelled.

    ```python
    @scheduler.on_future_cancelled
    def my_callback(future: Future):
        ...
    ```
    """
    on_finishing: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when the
    scheduler is finishing up.

    ```python
    @scheduler.on_finishing
    def my_callback():
        ...
    ```
    """
    on_finished: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    the scheduler finishes.

    ```python
    @scheduler.on_finished
    def my_callback():
        ...
    ```
    """
    on_stop: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when the
    scheduler is stopped.

    ```python
    @scheduler.on_stop
    def my_callback():
        ...
    ```
    """
    on_timeout: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when
    the scheduler reaches the timeout.

    ```python
    @scheduler.on_timeout
    def my_callback():
        ...
    ```
    """
    on_empty: Subscriber[[]]
    """A [`Subscriber`][amltk.events.Subscriber] which is called when the
    queue is empty.

    ```python
    @scheduler.on_empty
    def my_callback():
        ...
    ```
    """

    STARTED: Event[[]] = Event("on_start")
    FINISHING: Event[[]] = Event("on_finishing")
    FINISHED: Event[[]] = Event("on_finished")
    STOP: Event[[]] = Event("on_stop")
    TIMEOUT: Event[[]] = Event("on_timeout")
    EMPTY: Event[[]] = Event("on_empty")
    FUTURE_SUBMITTED: Event[Future] = Event("on_future_submitted")
    FUTURE_DONE: Event[Future] = Event("on_future_done")
    FUTURE_CANCELLED: Event[Future] = Event("on_future_cancelled")
    FUTURE_RESULT: Event[Future, Any] = Event("on_future_result")
    FUTURE_EXCEPTION: Event[Future, BaseException] = Event("on_future_exception")

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
        super().__init__()
        self.executor = executor
        self.unique_ref = f"Scheduler-{uuid4()}"
        self.emitter = Emitter()
        self.event_counts = self.emitter.event_counts

        # The current state of things and references to them
        self.queue = {}

        # Set up subscribers for events
        self.on_start = self.emitter.subscriber(self.STARTED)
        self.on_finishing = self.emitter.subscriber(self.FINISHING)
        self.on_finished = self.emitter.subscriber(self.FINISHED)
        self.on_stop = self.emitter.subscriber(self.STOP)
        self.on_timeout = self.emitter.subscriber(self.TIMEOUT)
        self.on_empty = self.emitter.subscriber(self.EMPTY)

        self.on_future_submitted = self.emitter.subscriber(self.FUTURE_SUBMITTED)
        self.on_future_done = self.emitter.subscriber(self.FUTURE_DONE)
        self.on_future_cancelled = self.emitter.subscriber(self.FUTURE_CANCELLED)
        self.on_future_exception = self.emitter.subscriber(self.FUTURE_EXCEPTION)
        self.on_future_result = self.emitter.subscriber(self.FUTURE_RESULT)

        self._terminate: Callable[[Executor], None] | None
        if terminate is True:
            self._terminate = termination_strategy(executor)
        else:
            self._terminate = terminate if callable(terminate) else None

        # This can be triggered either by `scheduler.stop` in a callback.
        # Has to be created inside the event loop so there's no issues
        self._stop_event: ContextEvent | None = None

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

        # A collection of things that want to register as being part of something
        # to render when the Scheduler is rendered.
        self._renderables: list[RenderableType] = [self.emitter]

        # These are extra user provided renderables during a call to `run()`. We
        # seperate these out so that we can remove them when the scheduler is
        # stopped.
        self._extra_renderables: list[RenderableType] | None = None

        # An indicator an object to render live output (if requested) with
        # `display=` on a call to `run()`
        self._live_output: Live | None = None

    @classmethod
    def with_processes(
        cls,
        max_workers: int | None = None,
        mp_context: BaseContext | Literal["fork", "spawn", "forkserver"] | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> Self:
        """Create a scheduler with a `ProcessPoolExecutor`.

        See [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor]
        for more details.
        """
        if isinstance(mp_context, str):
            from multiprocessing import get_context

            mp_context = get_context(mp_context)

        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
        )
        return cls(executor=executor)

    @classmethod
    def with_loky(  # noqa: PLR0913
        cls,
        max_workers: int | None = None,
        context: BaseContext | Literal["fork", "spawn", "forkserver"] | None = None,
        timeout: int = 10,
        kill_workers: bool = False,  # noqa: FBT002, FBT001
        reuse: bool | Literal["auto"] = "auto",
        job_reducers: Any | None = None,
        result_reducers: Any | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
        env: dict[str, str] | None = None,
    ) -> Self:
        """Create a scheduler with a `loky.get_reusable_executor`.

        See [loky documentation][https://loky.readthedocs.io/en/stable/API.html]
        for more details.
        """
        from loky import get_reusable_executor

        executor = get_reusable_executor(
            max_workers=max_workers,
            context=context,
            timeout=timeout,
            kill_workers=kill_workers,
            reuse=reuse,  # type: ignore
            job_reducers=job_reducers,
            result_reducers=result_reducers,
            initializer=initializer,
            initargs=initargs,
            env=env,
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
    ) -> Future[R]:
        """Submits a callable to be executed with the given arguments.

        Args:
            function: The callable to be executed as
                fn(*args, **kwargs) that returns a Future instance representing
                the execution of the callable.
            args: positional arguments to pass to the function
            kwargs: keyword arguments to pass to the function

        Raises:
            Scheduler.NotRunningError: If the scheduler is not running.

        Returns:
            A Future representing the given call.
        """
        if not self.running():
            msg = (
                f"Scheduler is not running, cannot submit task {function}"
                f" with {args=}, {kwargs=}"
            )
            raise SchedulerNotRunningError(msg)

        try:
            sync_future = self.executor.submit(function, *args, **kwargs)
            future = asyncio.wrap_future(sync_future)
        except Exception as e:
            logger.exception(f"Could not submit task {function}", exc_info=e)
            raise e

        self._register_future(future, function, *args, **kwargs)
        return future

    def task(
        self,
        function: Callable[P, R],
        *,
        plugins: TaskPlugin | Iterable[TaskPlugin] = (),
        init_plugins: bool = True,
    ) -> Task[P, R]:
        """Create a new task.

        Args:
            function: The function to run using the scheduler.
            plugins: The plugins to attach to the task.
            init_plugins: Whether to initialize the plugins.

        Returns:
            A new task.
        """
        task = Task(function, self, plugins=plugins, init_plugins=init_plugins)
        self.add_renderable(task)
        return task

    def _register_future(
        self,
        future: Future,
        function: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Registers the future into the queue and add a callback that will be called
        upon future completion. This callback will remove the future from the queue.
        """
        self.queue[future] = (function, args, kwargs)
        self._queue_has_items_event.set()

        self.on_future_submitted.emit(future)
        future.add_done_callback(self._register_complete)

        # Display if requested
        if self._live_output:
            self._live_output.refresh()
            future.add_done_callback(
                lambda _, live=self._live_output: live.refresh(),  # type: ignore
            )

    def _register_complete(self, future: Future) -> None:
        try:
            self.queue.pop(future)
        except ValueError as e:
            logger.error(
                f"{future=} was not found in the queue {self.queue}: {e}!",
                exc_info=True,
            )

        if future.cancelled():
            self.on_future_cancelled.emit(future)
            return

        self.on_future_done.emit(future)

        exception = future.exception()
        if exception:
            self.on_future_exception.emit(future, exception)
            if self._end_on_exception_flag and future.done():
                self.stop(stop_msg="Ending on first exception", exception=exception)
        else:
            result = future.result()
            self.on_future_result.emit(future, result)

    async def _monitor_queue_empty(self) -> None:
        """Monitor for the queue being empty and trigger an event when it is."""
        if not self.running():
            raise RuntimeError("The scheduler is not running!")

        while True:
            while self.queue:
                queue = list(self.queue)
                await asyncio.wait(queue, return_when=asyncio.ALL_COMPLETED)

            # Signal that the queue is now empty
            self._queue_has_items_event.clear()
            self.on_empty.emit()

            # Wait for an item to be in the queue
            await self._queue_has_items_event.wait()

            logger.debug("Queue has been filled again")

    async def _stop_when_triggered(self, stop_event: ContextEvent) -> bool:
        """Stop the scheduler when the stop event is set."""
        if not self.running():
            raise RuntimeError("The scheduler is not running!")

        await stop_event.wait()

        logger.debug("Stop event triggered, stopping scheduler")
        return True

    async def _run_scheduler(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
    ) -> ExitCode | BaseException:
        self.executor.__enter__()
        self._stop_event = ContextEvent()

        if self._live_output is not None:
            self._live_output.__enter__()

            # If we are doing a live display, we have to disable
            # warnings as they will screw up the display rendering
            # However, we re-enable it after the scheduler has finished running
            warning_catcher = warnings.catch_warnings()
            warning_catcher.__enter__()
            warnings.filterwarnings("ignore")
        else:
            warning_catcher = None

        # Declare we are running
        self._running_event.set()

        # Start a Thread Timer as our timing mechanism.
        # HACK: This is required because the SequentialExecutor mode
        # will not allow the async loop to run, meaning we can't update
        # any internal state.
        if timeout is not None:
            self._timeout_timer = Timer(timeout, lambda: None)
            self._timeout_timer.start()

        self.on_start.emit()

        # Monitor for `stop` being triggered
        stop_triggered = asyncio.create_task(
            self._stop_when_triggered(self._stop_event),
        )

        # Monitor for the queue being empty
        monitor_empty = asyncio.create_task(self._monitor_queue_empty())
        if end_on_empty:
            self.on_empty(lambda: monitor_empty.cancel(), hidden=True)

        # The timeout criterion is satisifed by the `timeout` arg
        await asyncio.wait(
            [stop_triggered, monitor_empty],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Determine the reason for stopping
        stop_reason: BaseException | Scheduler.ExitCode
        if stop_triggered.done() and self._stop_event.is_set():
            stop_reason = Scheduler.ExitCode.STOPPED

            msg, exception = self._stop_event.context
            _log = logger.exception if exception else logger.debug
            _log(f"Stop Message: {msg}", exc_info=exception)

            self.on_stop.emit()
            if self._end_on_exception_flag and exception:
                stop_reason = exception
            else:
                stop_reason = Scheduler.ExitCode.STOPPED
        elif monitor_empty.done():
            logger.debug("Scheduler stopped due to being empty.")
            stop_reason = Scheduler.ExitCode.EXHAUSTED
        elif timeout is not None:
            logger.debug(f"Scheduler stopping as {timeout=} reached.")
            stop_reason = Scheduler.ExitCode.TIMEOUT
            self.on_timeout.emit()
        else:
            logger.warning("Scheduler stopping for unknown reason!")
            stop_reason = Scheduler.ExitCode.UNKNOWN

        # Stop all runnings async tasks, i.e. monitoring the queue to trigger an event
        tasks = [monitor_empty, stop_triggered]
        for task in tasks:
            task.cancel()

        # Await all the cancelled tasks and read the exceptions
        await asyncio.gather(*tasks, return_exceptions=True)

        self.on_finishing.emit()
        logger.debug("Scheduler is finished")
        logger.debug(f"Shutting down scheduler executor with {wait=}")

        # The scheduler is now refusing jobs
        self._running_event.clear()
        logger.debug("Scheduler has shutdown and declared as no longer running")

        # This will try to end the tasks based on wait and self._terminate
        Scheduler._end_pending(
            wait=wait,
            futures=list(self.queue.keys()),
            executor=self.executor,
            termination_strategy=self._terminate,
        )

        self.on_finished.emit()
        logger.debug(f"Scheduler finished with status {stop_reason}")

        # Clear all events
        self._stop_event.clear()
        self._queue_has_items_event.clear()

        if self._live_output is not None:
            self._live_output.refresh()
            self._live_output.stop()

        if self._timeout_timer is not None:
            self._timeout_timer.cancel()

        if warning_catcher is not None:
            warning_catcher.__exit__()  # type: ignore

        return stop_reason

    def run(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
        on_exception: Literal["raise", "end", "ignore"] = "raise",
        asyncio_debug_mode: bool = False,
        display: bool | list[RenderableType] = False,
    ) -> ExitState:
        """Run the scheduler.

        Args:
            timeout: The maximum time to run the scheduler for in
                seconds. Defaults to `None` which means no timeout and it
                will end once the queue becomes empty.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.
            on_exception: What to do when an exception occurs.
                If "raise", the exception will be raised.
                If "ignore", the scheduler will continue running.
                If "end", the scheduler will end but not raise.
            asyncio_debug_mode: Whether to run the async loop in debug mode.
                Defaults to `False`. Please see [asyncio.run][] for more.
            display: Whether to display things in the console.
                If `True`, will display the scheduler and all its
                renderables. If a list of renderables, will display
                the scheduler itself plus those renderables.

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
                on_exception=on_exception,
                display=display,
            ),
            debug=asyncio_debug_mode,
        )

    async def async_run(
        self,
        *,
        timeout: float | None = None,
        end_on_empty: bool = True,
        wait: bool = True,
        on_exception: Literal["raise", "end", "ignore"] = "raise",
        display: bool | list[RenderableType] = False,
    ) -> ExitState:
        """Async version of `run`.

        Args:
            timeout: The maximum time to run the scheduler for.
                Defaults to `None` which means no timeout.
            end_on_empty: Whether to end the scheduler when the
                queue becomes empty. Defaults to `True`.
            wait: Whether to wait for the executor to shutdown.
            on_exception: Whether to end if an exception occurs.
                if "raise", the exception will be raised.
                If "ignore", the scheduler will continue running.
                If "end", the scheduler will end but not raise.
            display: Whether to display things in the console.
                If `True`, will display the scheduler and all its
                renderables. If a list of renderables, will display
                the scheduler itself plus those renderables.

        Returns:
            The reason for the scheduler ending.
        """
        if self.running():
            raise RuntimeError("Scheduler already seems to be running")

        logger.debug("Starting scheduler")

        # Make sure flags are set
        self._end_on_exception_flag.set(value=on_exception in ("raise", "end"))

        # If the user has requested to have a live display,
        # we will need to setup a `Live` instance to render to
        if display:
            from rich.live import Live

            if isinstance(display, list):
                self._extra_renderables = display

            self._live_output = Live(
                auto_refresh=False,
                get_renderable=self.__rich__,
            )

        loop = asyncio.get_running_loop()

        # Set the exception handler for asyncio
        previous_exception_handler = None
        if on_exception in ("raise", "end"):
            previous_exception_handler = loop.get_exception_handler()

            def custom_exception_handler(
                loop: asyncio.AbstractEventLoop,
                context: dict[str, Any],
            ) -> None:
                exception = context.get("exception")
                message = context.get("message")
                self.stop(stop_msg=message, exception=exception)

                # handle with previous handler
                if previous_exception_handler:
                    previous_exception_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(custom_exception_handler)

        # Run the actual scheduling loop
        result = await self._run_scheduler(
            timeout=timeout,
            end_on_empty=end_on_empty,
            wait=wait,
        )

        # Reset variables back to its default
        self._live_output = None
        self._extra_renderables = None
        self._end_on_exception_flag.reset()

        if previous_exception_handler is not None:
            loop.set_exception_handler(previous_exception_handler)

        # If we were meant to end on an exception and the result
        # we got back from the scheduler was an exception, raise it
        if isinstance(result, BaseException):
            if on_exception == "raise":
                raise result

            return ExitState(code=Scheduler.ExitCode.EXCEPTION, exception=result)

        return ExitState(code=result)

    run_in_notebook = async_run
    """Alias for [`async_run()`][amltk.Scheduler.async_run]"""

    def stop(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Stop the scheduler.

        The scheduler will stop, finishing currently running tasks depending
        on the `wait=` parameter to [`Scheduler.run`][amltk.Scheduler.run].

        The call signature is kept open with `*args, **kwargs` to make it
        easier to include in any callback.

        Args:
            *args: Logged in a debug message
            **kwargs: Logged in a debug message

                * **stop_msg**: The message to pass to the stop event which
                    gets logged as the stop reason.

                * **exception**: The exception to pass to the stop event which
                    gets logged as the stop reason.
        """
        if not self.running():
            return

        assert self._stop_event is not None

        msg = kwargs.get("stop_msg", "stop() called")

        self._stop_event.set(msg=f"{msg}", exception=kwargs.get("exception"))
        self._running_event.clear()

    @staticmethod
    def _end_pending(
        *,
        futures: list[Future],
        executor: Executor,
        wait: bool = True,
        termination_strategy: Callable[[Executor], Any] | None = None,
    ) -> None:
        if wait:
            logger.debug("Waiting for currently running tasks to finish.")
            executor.shutdown(wait=wait)
        elif termination_strategy is None:
            logger.warning(
                "Cancelling currently running tasks and then waiting "
                f" as there is no termination strategy provided for {executor=}`.",
            )
            # Just try to cancel the tasks. Will cancel pending tasks
            # but executors like dask will even kill the job
            for future in futures:
                if not future.done():
                    logger.debug(f"Cancelling {future=}")
                    future.cancel()

            # Here we wait, if we could  cancel, then we wait for that
            # to happen, otherwise we are just waiting as anticipated.
            executor.shutdown(wait=wait)
        else:
            logger.debug(f"Terminating workers with {termination_strategy=}")
            for future in futures:
                if not future.done():
                    logger.debug(f"Cancelling {future=}")
                    future.cancel()
            termination_strategy(executor)
            executor.shutdown(wait=wait)

    def add_renderable(self, renderable: RenderableType) -> None:
        """Add a renderable to the scheduler.

        This will be displayed whenever the scheduler is displayed.
        """
        self._renderables.append(renderable)

    def __rich__(self) -> RenderableType:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Column, Table
        from rich.text import Text
        from rich.tree import Tree

        from amltk.richutil import richify
        from amltk.richutil.renderers.function import Function

        MAX_FUTURE_ITEMS = 5
        OFFSETS = 1 + 1 + 2  # Header + ellipses space + panel borders

        title = Text("Scheduler", style="magenta bold")
        if self.running():
            title.append(" (running)", style="green")

        future_table = Table.grid()

        # Select the most latest items
        future_items = list(self.queue.items())[-MAX_FUTURE_ITEMS:]
        for future, (func, args, kwargs) in future_items:
            entry = Function(
                func,
                (args, kwargs),
                link=False,
                prefix=future._state,
                no_wrap=True,
            )
            future_table.add_row(entry)

        if len(self.queue) > MAX_FUTURE_ITEMS:
            future_table.add_row(Text("...", style="yellow"))

        queue_column_text = Text.assemble(
            "Queue: (",
            (f"{len(self.queue)}", "yellow"),
            ")",
        )

        layout_table = Table(
            Column("Executor", ratio=1),
            Column(queue_column_text, ratio=2),
            box=None,
            expand=True,
            padding=(0, 1),
        )
        layout_table.add_row(richify(self.executor), future_table)

        title = Panel(
            layout_table,
            title=title,
            title_align="left",
            border_style="magenta",
            height=MAX_FUTURE_ITEMS + OFFSETS,
        )
        tree = Tree(title, guide_style="magenta bold")

        for renderable in self._renderables:
            tree.add(renderable)

        if not self._extra_renderables:
            return tree

        return Group(tree, *self._extra_renderables)

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

        EXCEPTION = auto()
        """The scheduler finished because of an exception."""
