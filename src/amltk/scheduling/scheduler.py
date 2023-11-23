"""The [`Scheduler`][amltk.scheduling.Scheduler] uses
an [`Executor`][concurrent.futures.Executor], a builtin python native with
a `#!python submit(f, *args, **kwargs)` function to submit compute to
be compute else where, whether it be locally or remotely.

The `Scheduler` is primarily used to dispatch compute to an `Executor` and
emit `@events`, which can trigger user callbacks.

Typically you should not use the `Scheduler` directly for dispatching and
responding to computed functions, but rather use a [`Task`][amltk.scheduling.Task]

??? note "Running in a Jupyter Notebook/Colab"

    If you are using a Jupyter Notebook, you likley need to use the following
    at the top of your notebook:

    ```python
    import nest_asyncio  # Only necessary in Notebooks
    nest_asyncio.apply()

    scheduler.run(...)
    ```

    This is due to the fact a notebook runs in an async context. If you do not
    wish to use the above snippet, you can instead use:

    ```python
    await scheduler.async_run(...)
    ```

??? tip "Basic Usage"

    In this example, we create a scheduler that uses local processes as
    workers. We then create a task that will run a function `fn` and submit it
    to the scheduler. Lastly, a callback is registered to `@on_future_result` to print the
    result when the compute is done.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler

    def fn(x: int) -> int:
        return x + 1
    from amltk._doc import make_picklable; make_picklable(fn)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)

    @scheduler.on_start
    def launch_the_compute():
        scheduler.submit(fn, 1)

    @scheduler.on_future_result
    def callback(future, result):
        print(f"Result: {result}")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler)  # markdown-exec: hide
    ```

    The last line in the previous example called
    [`scheduler.run()`][amltk.scheduling.Scheduler.run] is what starts the scheduler
    running, in which it will first emit the `@on_start` event. This triggered the
    callback `launch_the_compute()` which submitted the function `fn` with the
    arguments `#!python 1`.

    The scheduler then ran the compute and waited for it to complete, emitting the
    `@on_future_result` event when it was done successfully. This triggered the callback
    `callback()` which printed the result.

    At this point, there is no more compute happening and no more events to respond to
    so the scheduler will halt.

??? example "`@events`"

    === "Scheduler Status Events"

        When the scheduler enters some important state, it will emit an event
        to let you know.

        === "`@on_start`"

            ::: amltk.scheduling.Scheduler.on_start

        === "`@on_finishing`"

            ::: amltk.scheduling.Scheduler.on_finishing

        === "`@on_finished`"

            ::: amltk.scheduling.Scheduler.on_finished

        === "`@on_stop`"

            ::: amltk.scheduling.Scheduler.on_stop

        === "`@on_timeout`"

            ::: amltk.scheduling.Scheduler.on_timeout

        === "`@on_empty`"

            ::: amltk.scheduling.Scheduler.on_empty

    === "Submitted Compute Events"

        When any compute goes through the `Scheduler`, it will emit an event
        to let you know. You should however prefer to use a
        [`Task`][amltk.scheduling.Task] as it will emit specific events
        for the task at hand, and not all compute.

        === "`@on_future_submitted`"

            ::: amltk.scheduling.Scheduler.on_future_submitted

        === "`@on_future_result`"

            ::: amltk.scheduling.Scheduler.on_future_result

        === "`@on_future_exception`"

            ::: amltk.scheduling.Scheduler.on_future_exception

        === "`@on_future_done`"

            ::: amltk.scheduling.Scheduler.on_future_done

        === "`@on_future_cancelled`"

            ::: amltk.scheduling.Scheduler.on_future_cancelled


??? tip "Common usages of `run()`"

    There are various ways to [`run()`][amltk.scheduling.Scheduler.run] the
    scheduler, notably how long it should run with `timeout=` and also how
    it should react to any exception that may have occurred within the `Scheduler`
    itself or your callbacks.

    Please see the [`run()`][amltk.scheduling.Scheduler.run] API doc for more
    details and features, however we show two common use cases of using the `timeout=`
    parameter.

    You can render a live display using [`run(display=...)`][amltk.scheduling.Scheduler.run].
    This require [`rich`](https://github.com/Textualize/rich) to be installed. You
    can install this with `#!bash pip install rich` or `#!bash pip install amltk[rich]`.


    === "`run(timeout=...)`"

        You can tell the `Scheduler` to stop after a certain amount of time
        with the `timeout=` argument to [`run()`][amltk.scheduling.Scheduler.run].

        This will also trigger the `@on_timeout` event as seen in the `Scheduler` output.

        ```python exec="true" source="material-block" html="True" hl_lines="19"
        import time
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(1)

        def expensive_function() -> int:
            time.sleep(0.1)
            return 42
        from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

        @scheduler.on_start
        def submit_calculations() -> None:
            scheduler.submit(expensive_function)

        # The will endlessly loop the scheduler
        @scheduler.on_future_done
        def submit_again(future: Future) -> None:
            if scheduler.running():
                scheduler.submit(expensive_function)

        scheduler.run(timeout=1)  # End after 1 second
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

    === "`run(timeout=..., wait=False)`"

        By specifying that the `Scheduler` should not wait for ongoing tasks
        to finish, the `Scheduler` will attempt to cancel and possibly terminate
        any running tasks.

        ```python exec="true" source="material-block" html="True"
        import time
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(1)

        def expensive_function() -> None:
            time.sleep(10)

        from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

        @scheduler.on_start
        def submit_calculations() -> None:
            scheduler.submit(expensive_function)

        scheduler.run(timeout=1, wait=False)  # End after 1 second
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

        ??? info "Forcibly Terminating Workers"

            As an `Executor` does not provide an interface to forcibly
            terminate workers, we provide `Scheduler(terminate=...)` as a custom
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
            However there is no guarantee of this and is up to how the
            `Executor` handles this.

??? example "Scheduling something to be run later"

    You can schedule some function to be run later using the
    [`#!python scheduler.call_later()`][amltk.scheduling.Scheduler.call_later] method.

    !!! note

        This does not run the function in the background, it just schedules some
        function to be called later, where you could perhaps then use submit to
        scheduler a [`Task`][amltk.scheduling.Task] to run the function in the
        background.

    ```python exec="true" source="material-block" result="python"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    def fn() -> int:
        print("Ending now!")
        scheduler.stop()

    @scheduler.on_start
    def schedule_fn() -> None:
        scheduler.call_later(1, fn)

    scheduler.run(end_on_empty=False)
    ```

"""  # noqa: E501
from __future__ import annotations

import asyncio
import logging
import warnings
from asyncio import Future
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from threading import Timer
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)
from typing_extensions import Self, override
from uuid import uuid4

from amltk._asyncm import ContextEvent
from amltk._functional import Flag
from amltk._richutil.renderable import RichRenderable
from amltk.exceptions import SchedulerNotRunningError
from amltk.scheduling.events import Emitter, Event, Subscriber
from amltk.scheduling.executors import SequentialExecutor
from amltk.scheduling.task import Task
from amltk.scheduling.termination_strategies import termination_strategy

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from rich.console import RenderableType
    from rich.live import Live

    from amltk.scheduling.executors.dask_jobqueue import DJQ_NAMES
    from amltk.scheduling.plugins import Plugin
    from amltk.scheduling.plugins.comm import Comm

    P = ParamSpec("P")
    R = TypeVar("R")


logger = logging.getLogger(__name__)


class Scheduler(RichRenderable):
    """A scheduler for submitting tasks to an Executor."""

    executor: Executor
    """The executor to use to run tasks."""

    emitter: Emitter
    """The emitter to use for events."""

    queue: dict[Future, tuple[Callable, tuple, dict]]
    """The queue of tasks running."""

    on_start: Subscriber[[]]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when the
    scheduler starts. This is the first event emitted by the scheduler and
    one of the only ways to submit the initial compute to the scheduler.

    ```python
    @scheduler.on_start
    def my_callback():
        ...
    ```
    """
    on_future_submitted: Subscriber[Future]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    some compute is submitted.

    ```python
    @scheduler.on_future_submitted
    def my_callback(future: Future):
        ...
    ```
    """
    on_future_done: Subscriber[Future]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    some compute is done, regardless of whether it was successful or not.

    ```python
    @scheduler.on_future_done
    def my_callback(future: Future):
        ...
    ```
    """
    on_future_result: Subscriber[Future, Any]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    a future returned with a result, no exception raise.

    ```python
    @scheduler.on_future_result
    def my_callback(future: Future, result: Any):
        ...
    ```
    """
    on_future_exception: Subscriber[Future, BaseException]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    some compute raised an uncaught exception.

    ```python
    @scheduler.on_future_exception
    def my_callback(future: Future, exception: BaseException):
        ...
    ```
    """
    on_future_cancelled: Subscriber[Future]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called
    when a future is cancelled. This usually occurs due to the underlying Scheduler,
    and is not something we do directly, other than when shutting down the scheduler.

    ```python
    @scheduler.on_future_cancelled
    def my_callback(future: Future):
        ...
    ```
    """
    on_finishing: Subscriber[[]]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when the
    scheduler is finishing up. This occurs right before the scheduler shuts down
    the executor.

    ```python
    @scheduler.on_finishing
    def my_callback():
        ...
    ```
    """
    on_finished: Subscriber[[]]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    the scheduler is finished, has shutdown the executor and possibly
    terminated any remaining compute.

    ```python
    @scheduler.on_finished
    def my_callback():
        ...
    ```
    """
    on_stop: Subscriber[str, BaseException | None]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when the
    scheduler is has been stopped due to the [`stop()`][amltk.scheduling.Scheduler.stop]
    method being called.

    ```python
    @scheduler.on_stop
    def my_callback(stop_msg: str, exception: BaseException | None):
        ...
    ```
    """
    on_timeout: Subscriber[[]]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when
    the scheduler reaches the timeout.

    ```python
    @scheduler.on_timeout
    def my_callback():
        ...
    ```
    """
    on_empty: Subscriber[[]]
    """A [`Subscriber`][amltk.scheduling.events.Subscriber] which is called when the
    queue is empty. This can be useful to re-fill the queue and prevent the
    scheduler from exiting.

    ```python
    @scheduler.on_empty
    def my_callback():
        ...
    ```
    """

    STARTED: Event[[]] = Event("on_start")
    FINISHING: Event[[]] = Event("on_finishing")
    FINISHED: Event[[]] = Event("on_finished")
    STOP: Event[str, BaseException | None] = Event("on_stop")
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
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a SLURM cluster.

        This is useful for running on a SLURM cluster. Uses
        [dask_jobqueue.SLURMCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_pbs(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a PBS cluster.

        This is useful for running on a PBS cluster. Uses
        [dask_jobqueue.PBSCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_sge(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a SGE cluster.

        This is useful for running on a SGE cluster. Uses
        [dask_jobqueue.SGECluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_oar(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a OAR cluster.

        This is useful for running on a OAR cluster. Uses
        [dask_jobqueue.OARCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_moab(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a Moab cluster.

        This is useful for running on a Moab cluster. Uses
        [dask_jobqueue.MoabCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_lsf(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a LSF cluster.

        This is useful for running on a LSF cluster. Uses
        [dask_jobqueue.LSFCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_htcondor(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler that runs on a HTCondor cluster.

        This is useful for running on a HTCondor cluster. Uses
        [dask_jobqueue.HTCondorCluster][].

        Args:
            n_workers: The number of workers to start.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            adaptive=adaptive,
            submit_command=submit_command,
            cancel_command=cancel_command,
            **kwargs,
        )

    @classmethod
    def with_dask_jobqueue(
        cls,
        name: DJQ_NAMES,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Scheduler with using `dask-jobqueue`.

        See [`dask_jobqueue`][dask_jobqueue] for more details.

        [dask_jobqueue]: https://jobqueue.dask.org/en/latest/

        Args:
            name: The name of the jobqueue to use. This is the name of the
                class in `dask_jobqueue` to use. For example, to use
                `dask_jobqueue.SLURMCluster`, you would use `slurm`.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed
                allocate all workers. This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers
                specified.
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
            from amltk.scheduling.executors.dask_jobqueue import DaskJobqueueExecutor

        except ImportError as e:
            raise ImportError(
                f"To use the {name} executor, you must install the "
                "`dask-jobqueue` package.",
            ) from e

        executor = DaskJobqueueExecutor.from_str(
            name,
            n_workers=n_workers,
            adaptive=adaptive,
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
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        """Submits a callable to be executed with the given arguments.

        Args:
            fn: The callable to be executed as
                fn(*args, **kwargs) that returns a Future instance representing
                the execution of the callable.
            args: positional arguments to pass to the function
            kwargs: keyword arguments to pass to the function

        Raises:
            SchedulerNotRunningError: If the scheduler is not running.
                You can protect against this using,
                [`scheduler.running()`][amltk.scheduling.scheduler.Scheduler.running].

        Returns:
            A Future representing the given call.
        """
        if not self.running():
            msg = (
                f"Scheduler is not running, cannot submit task {fn}"
                f" with {args=}, {kwargs=}"
            )
            raise SchedulerNotRunningError(msg)

        try:
            sync_future = self.executor.submit(fn, *args, **kwargs)
            future = asyncio.wrap_future(sync_future)
        except Exception as e:
            logger.exception(f"Could not submit task {fn}", exc_info=e)
            raise e

        self._register_future(future, fn, *args, **kwargs)
        return future

    @overload
    def task(
        self,
        function: Callable[Concatenate[Comm, P], R],
        *,
        plugins: Comm.Plugin | Iterable[Comm.Plugin | Plugin] = ...,
        init_plugins: bool = ...,
    ) -> Task[P, R]:
        ...

    @overload
    def task(
        self,
        function: Callable[P, R],
        *,
        plugins: Plugin | Iterable[Plugin] = (),
        init_plugins: bool = True,
    ) -> Task[P, R]:
        ...

    def task(
        self,
        function: Callable[P, R] | Callable[Concatenate[Comm, P], R],
        *,
        plugins: Plugin | Iterable[Plugin] = (),
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
        # HACK: Not that the type: ignore is due to the fact that we can't use type
        # checking to enforce that
        # A. `function` is a callable with the first arg being a Comm
        # B. `plugins`
        task = Task(function, self, plugins=plugins, init_plugins=init_plugins)  # type: ignore
        self.add_renderable(task)
        return task  # type: ignore

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
        # Display if requested
        if self._live_output:
            self._live_output.refresh()
            future.add_done_callback(
                lambda _, live=self._live_output: live.refresh(),  # type: ignore
            )

        future.add_done_callback(self._register_complete)

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
    ) -> ExitState.Code | BaseException:
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

        # The timeout criterion is satisfied by the `timeout` arg
        await asyncio.wait(
            [stop_triggered, monitor_empty],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Determine the reason for stopping
        stop_reason: BaseException | ExitState.Code
        if stop_triggered.done() and self._stop_event.is_set():
            stop_reason = ExitState.Code.STOPPED

            msg, exception = self._stop_event.context
            _log = logger.exception if exception else logger.debug
            _log(f"Stop Message: {msg}", exc_info=exception)

            self.on_stop.emit(str(msg), exception)
            if self._end_on_exception_flag and exception:
                stop_reason = exception
            else:
                stop_reason = ExitState.Code.STOPPED
        elif monitor_empty.done():
            logger.debug("Scheduler stopped due to being empty.")
            stop_reason = ExitState.Code.EXHAUSTED
        elif timeout is not None:
            logger.debug(f"Scheduler stopping as {timeout=} reached.")
            stop_reason = ExitState.Code.TIMEOUT
            self.on_timeout.emit()
        else:
            logger.warning("Scheduler stopping for unknown reason!")
            stop_reason = ExitState.Code.UNKNOWN

        # Stop all running async tasks, i.e. monitoring the queue to trigger an event
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
                will end once the queue is empty if `end_on_empty=True`.
            end_on_empty: Whether to end the scheduler when the queue becomes empty.
            wait: Whether to wait for currently running compute to finish once
                the `Scheduler` is shutting down.

                * If `#!python True`, will wait for all currently running compute.
                * If `#!python False`, will attempt to cancel/terminate all currently
                    running compute and shutdown the executor. This may be useful
                    if you want to end the scheduler as quickly as possible or
                    respect the `timeout=` more precisely.
            on_exception: What to do when an exception occurs in the scheduler
                or callbacks (**Does not apply to submitted compute!**)

                * If `#!python "raise"`, the scheduler will stop and raise the
                    exception at the point where you called `run()`.
                * If `#!python "ignore"`, the scheduler will continue running,
                    ignoring the exception. This may be useful when requiring more
                    robust execution.
                * If `#!python "end"`, similar to `#!python "raise"`, the scheduler
                    will stop but no exception will occur and the control flow
                    will return gracefully to the point where you called `run()`.
            asyncio_debug_mode: Whether to run the async loop in debug mode.
                Defaults to `False`. Please see [asyncio.run][] for more.
            display: Whether to display the scheduler live in the console.

                * If `#!python True`, will display the scheduler and all its tasks.
                * If a `#!python list[RenderableType]` , will display the scheduler
                    itself plus those renderables.

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

        This can be useful if you are already running in an async context,
        such as in a web server or Jupyter notebook.

        Please see [`run()`][amltk.Scheduler.run] for more details.
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

                # handle with previous handler
                if previous_exception_handler:
                    previous_exception_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

                self.stop(stop_msg=message, exception=exception)

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

            return ExitState(code=ExitState.Code.EXCEPTION, exception=result)

        return ExitState(code=result)

    def stop(
        self,
        *args: Any,
        stop_msg: str | None = None,
        exception: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        """Stop the scheduler.

        The scheduler will stop, finishing currently running tasks depending
        on the `wait=` parameter to [`Scheduler.run`][amltk.Scheduler.run].

        The call signature is kept open with `*args, **kwargs` to make it
        easier to include in any callback.

        Args:
            *args: Logged in a debug message
            **kwargs: Logged in a debug message
            stop_msg: The message to log when stopping the scheduler.
            exception: The exception which incited `stop()` to be called.
                Will be used by the `Scheduler` to possibly raise the exception
                to the user.
        """
        if not self.running():
            return

        assert self._stop_event is not None

        msg = stop_msg if stop_msg is not None else "scheduler.stop() was called."
        logger.debug(f"Stopping scheduler: {msg} {args=} {kwargs=}")

        self._stop_event.set(msg=msg, exception=exception)
        self._running_event.clear()

    def call_later(
        self,
        delay: float,
        fn: Callable[P, Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> asyncio.TimerHandle:
        """Schedule a function to be run after a delay.

        Args:
            delay: The delay in seconds.
            fn: The function to run.
            args: The positional arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.

        Returns:
            A timer handle that can be used to cancel the function.
        """
        if not self.running():
            raise RuntimeError("Scheduler is not running!")

        _fn = partial(fn, *args, **kwargs)
        loop = asyncio.get_running_loop()
        return loop.call_later(delay, _fn)

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
        """Add a renderable object to the scheduler.

        This will be displayed whenever the scheduler is displayed.
        """
        self._renderables.append(renderable)

    @override
    def __rich__(self) -> RenderableType:
        from rich.console import Group
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Column, Table
        from rich.text import Text
        from rich.tree import Tree

        from amltk._richutil import richify
        from amltk._richutil.renderers.function import Function

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
        layout_table.add_row(richify(self.executor, otherwise=Pretty), future_table)

        panel = Panel(
            layout_table,
            title=title,
            title_align="left",
            border_style="magenta",
            height=MAX_FUTURE_ITEMS + OFFSETS,
        )
        tree = Tree(panel, guide_style="magenta bold")

        for renderable in self._renderables:
            tree.add(renderable)

        if not self._extra_renderables:
            return tree

        return Group(tree, *self._extra_renderables)


@dataclass
class ExitState:
    """The exit state of a scheduler.

    Attributes:
        reason: The reason for the exit.
        exception: The exception that caused the exit, if any.
    """

    code: ExitState.Code
    exception: BaseException | None = None

    class Code(Enum):
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
