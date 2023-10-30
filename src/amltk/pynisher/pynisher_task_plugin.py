"""A plugin that wraps a task in a pynisher to enforce limits on it.

Please note that this plugin requires the `pynisher` package to be installed.

Documentation for the `pynisher` package can be found [here](https://github.com/automl/pynisher).

Notably, there are some limitations to the `pynisher` package with Mac and Windows
which are [listed here](https://github.com/automl/pynisher#features).
"""
from __future__ import annotations

from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Callable, TypeVar
from typing_extensions import ParamSpec, Self, override

from amltk.events import Event
from amltk.scheduling.task_plugin import TaskPlugin
from pynisher import Pynisher

if TYPE_CHECKING:
    import asyncio

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")


class PynisherPlugin(TaskPlugin):
    """A plugin that wraps a task in a pynisher to enforce limits on it.

    This plugin wraps a task function in a `Pynisher` instance to enforce
    limits on the task. The limits are set by any of `memory_limit=`,
    `cpu_time_limit=` and `wall_time_limit=`.

    Adds four new events to the task

    * [`TIMEOUT`][amltk.pynisher.PynisherPlugin.TIMEOUT]
        - subscribe with `@task.on("pynisher-timeout")`
    * [`MEMORY_LIMIT_REACHED`][amltk.pynisher.PynisherPlugin.MEMORY_LIMIT_REACHED]
        - subscribe with `@task.on("pynisher-memory-limit")`
    * [`CPU_TIME_LIMIT_REACHED`][amltk.pynisher.PynisherPlugin.CPU_TIME_LIMIT_REACHED]
        - subscribe with `@task.on("pynisher-cputime-limit")`
    * [`WALL_TIME_LIMIT_REACHED`][amltk.pynisher.PynisherPlugin.WALL_TIME_LIMIT_REACHED]
        - subscribe with `@task.on("pynisher-walltime-limit")`


    ```python exec="true" source="material-block" result="python" title="PynisherPlugin"
    from amltk.scheduling import Task, Scheduler
    from amltk.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        time.sleep(x)
        return "yay"

    scheduler = Scheduler.with_sequential()
    task = scheduler.task(f, plugins=PynisherPlugin(wall_time_limit=(1, "s")))

    @scheduler.on_start
    def on_start():
        task(3)

    @task.on("pynisher-wall-time-limit")
    def on_wall_time_limit(exception):
        print(f"Wall time limit reached!")

    end_status = scheduler.run(on_exception="end")
    print(end_status)
    ```

    Attributes:
        memory_limit: The memory limit of the task.
        cpu_time_limit: The cpu time limit of the task.
        wall_time_limit: The wall time limit of the task.
    """

    name = "pynisher-plugin"
    """The name of the plugin."""

    TIMEOUT: Event[Pynisher.TimeoutException] = Event("pynisher-timeout")
    """A Task timed out.

    Will call any subscribers with the exception as the argument.

    ```python
    @task.on("pynisher-timeout")
    def on_timeout(exception: PynisherPlugin.TimeoutException):
        ...
    ```
    """

    MEMORY_LIMIT_REACHED: Event[Pynisher.MemoryLimitException] = Event(
        "pynisher-memory-limit",
    )
    """A Task was submitted but reached it's memory limit.

    Will call any subscribers with the exception as the argument.

    ```python
    @task.on("pynisher-memory-limit")
    def on_memout(exception: PynisherPlugin.MemoryLimitException):
        ...
    ```
    """

    CPU_TIME_LIMIT_REACHED: Event[Pynisher.CpuTimeoutException] = Event(
        "pynisher-cpu-time-limit",
    )
    """A Task was submitted but reached it's cpu time limit.

    Will call any subscribers with the exception as the argument.

    ```python
    @task.on("pynisher-cpu-time-limit")
    def on_cpu_time_limit(exception: PynisherPlugin.TimeoutException):
        ...
    ```
    """

    WALL_TIME_LIMIT_REACHED: Event[Pynisher.WallTimeoutException] = Event(
        "pynisher-wall-time-limit",
    )
    """A Task was submitted but reached it's wall time limit.

    Will call any subscribers with the exception as the argument.

    ```python
    @task.on("pynisher-wall-time-limit")
    def on_wall_time_limit(exception: PynisherPlugin.TimeoutException):
        ...
    ```
    """

    TimeoutException = Pynisher.TimeoutException
    """The exception that is raised when a task times out."""

    MemoryLimitException = Pynisher.MemoryLimitException
    """The exception that is raised when a task reaches it's memory limit."""

    CpuTimeoutException = Pynisher.CpuTimeoutException
    """The exception that is raised when a task reaches it's cpu time limit."""

    WallTimeoutException = Pynisher.WallTimeoutException
    """The exception that is raised when a task reaches it's wall time limit."""

    def __init__(
        self,
        *,
        memory_limit: int | tuple[int, str] | None = None,
        cpu_time_limit: int | tuple[float, str] | None = None,
        wall_time_limit: int | tuple[float, str] | None = None,
        context: BaseContext | None = None,
    ):
        """Initialize a `PynisherPlugin` instance.

        Args:
            memory_limit: The memory limit to wrap the task in. Base unit is in bytes
                but you can specify `(value, unit)` where `unit` is one of
                `("B", "KB", "MB", "GB")`. Defaults to `None`
            cpu_time_limit: The cpu time limit to wrap the task in. Base unit is in
                seconds but you can specify `(value, unit)` where `unit` is one of
                `("s", "m", "h")`. Defaults to `None`
            wall_time_limit: The wall time limit for the task. Base unit is in seconds
                but you can specify `(value, unit)` where `unit` is one of
                `("s", "m", "h")`. Defaults to `None`.
            context: The context to use for multiprocessing. Defaults to `None`.
                See [`multiprocessing.get_context()`][multiprocessing.get_context]
        """
        super().__init__()
        self.memory_limit = memory_limit
        self.cpu_time_limit = cpu_time_limit
        self.wall_time_limit = wall_time_limit
        self.context = context

        self.task: Task

    @override
    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict]:
        """Wrap a task function in a `Pynisher` instance."""
        # If any of our limits is set, we need to wrap it in Pynisher
        # to enfore these limits.
        if any(
            limit is not None
            for limit in (self.memory_limit, self.cpu_time_limit, self.wall_time_limit)
        ):
            fn = Pynisher(
                fn,
                memory=self.memory_limit,
                cpu_time=self.cpu_time_limit,
                wall_time=self.wall_time_limit,
                terminate_child_processes=True,
                context=self.context,
            )

        return fn, args, kwargs

    @override
    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task
        task.emitter.add_event(
            self.TIMEOUT,
            self.MEMORY_LIMIT_REACHED,
            self.CPU_TIME_LIMIT_REACHED,
            self.WALL_TIME_LIMIT_REACHED,
        )

        # Check the exception and emit pynisher specific ones too
        task.on_exception(self._check_to_emit_pynisher_exception, hidden=True)

    @override
    def copy(self) -> Self:
        """Return a copy of the plugin.

        Please see [`TaskPlugin.copy()`][amltk.TaskPlugin.copy].
        """
        return self.__class__(
            memory_limit=self.memory_limit,
            cpu_time_limit=self.cpu_time_limit,
            wall_time_limit=self.wall_time_limit,
        )

    def _check_to_emit_pynisher_exception(
        self,
        _: asyncio.Future,
        exception: BaseException,
    ) -> None:
        """Check if the exception is a pynisher exception and emit it."""
        if isinstance(exception, Pynisher.CpuTimeoutException):
            self.task.emitter.emit(self.TIMEOUT, exception)
            self.task.emitter.emit(self.CPU_TIME_LIMIT_REACHED, exception)
        elif isinstance(exception, self.WallTimeoutException):
            self.task.emitter.emit(self.TIMEOUT)
            self.task.emitter.emit(self.WALL_TIME_LIMIT_REACHED, exception)
        elif isinstance(exception, self.MemoryLimitException):
            self.task.emitter.emit(self.MEMORY_LIMIT_REACHED, exception)
