"""A plugin that wraps a task in a pynisher to enforce limits on it.

Please note that this plugin requires the `pynisher` package to be installed.

Documentation for the `pynisher` package can be found [here](https://github.com/automl/pynisher).

Notably, there are some limitations to the `pynisher` package with Mac and Windows
which are [listed here](https://github.com/automl/pynisher#features).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from byop.events import Event
from byop.scheduling.task_plugin import TaskPlugin
from pynisher import Pynisher

if TYPE_CHECKING:
    from byop.events import Subscriber
    from byop.scheduling.task import Task


class PynisherPlugin(TaskPlugin):
    """A plugin that wraps a task in a pynisher to enforce limits on it.

    This plugin wraps a task function in a `Pynisher` instance to enforce
    limits on the task. The limits are set by any of `memory_limit=`,
    `cpu_time_limit=` and `wall_time_limit=`.

    ```python exec="true" source="material-block" result="python" title="PynisherPlugin"
    from byop.scheduling import Task, Scheduler
    from byop.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        time.sleep(x)
        return "yay"

    scheduler = Scheduler.with_sequential()

    pynisher = PynisherPlugin(wall_time_limit=(1, "s"))
    task = Task(f, scheduler, plugins=[pynisher])

    @scheduler.on_start
    def on_start():
        task(3)

    @pynisher.on_wall_time_limit_reached
    def on_wall_time_limit(exception):
        print(f"Wall time limit reached!")

    scheduler.run(timeout=5, raises=False)
    ```

    Attributes:
        memory_limit: The memory limit of the task.
        cpu_time_limit: The cpu time limit of the task.
        wall_time_limit: The wall time limit of the task.
        on_timeout: A subscriber to the
            [`TIMEOUT`][byop.pynisher.PynisherPlugin.TIMEOUT] event. This is triggered
            for both `WALL_TIME_LIMIT_REACHED` and `CPU_TIME_LIMIT_REACHED`.
        on_memory_limit_reached: A subscriber to the
            [`MEMORY_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.MEMORY_LIMIT_REACHED]
            event.
        on_cpu_time_limit_reached: A subscriber to the
            [`CPU_TIME_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.CPU_TIME_LIMIT_REACHED]
            event.
        on_wall_time_limit_reached: A subscriber to the
            [`WALL_TIME_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.WALL_TIME_LIMIT_REACHED]
            event.
    """

    name = "pynisher-plugin"
    """The name of the plugin."""

    TIMEOUT: Event[BaseException] = Event("pynisher-timeout")
    """A Task timed out."""

    MEMORY_LIMIT_REACHED: Event[BaseException] = Event("pynisher-memory-limit")
    """A Task was submitted but reached it's memory limit."""

    CPU_TIME_LIMIT_REACHED: Event[BaseException] = Event("pynisher-cputime-limit")
    """A Task was submitted but reached it's cpu time limit."""

    WALL_TIME_LIMIT_REACHED: Event[BaseException] = Event("pynisher-walltime-limit")
    """A Task was submitted but reached it's wall time limit."""

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
    ):
        """Initialize a `PynisherPlugin` instance.

        Args:
            memory_limit: The memory limit to wrap the task in. Defaults to `None`
            cpu_time_limit: The cpu time limit to wrap the task in. Defaults to `None`
            wall_time_limit: The wall time limit for the task. Defaults to `None`.
        """
        self.memory_limit = memory_limit
        self.cpu_time_limit = cpu_time_limit
        self.wall_time_limit = wall_time_limit

        self.task: Task

        self.on_timeout: Subscriber[BaseException]
        self.on_memory_limit_reached: Subscriber[BaseException]
        self.on_cpu_time_limit_reached: Subscriber[BaseException]
        self.on_wall_time_limit_reached: Subscriber[BaseException]

    def wrap(self, fn: Callable) -> Callable:
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
            )

        return fn

    def attach(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

        # Register our own subscribers
        self.on_timeout = task.subscriber(self.TIMEOUT)
        self.on_wall_time_limit_reached = task.subscriber(self.WALL_TIME_LIMIT_REACHED)
        self.on_cpu_time_limit_reached = task.subscriber(self.CPU_TIME_LIMIT_REACHED)
        self.on_memory_limit_reached = task.subscriber(self.MEMORY_LIMIT_REACHED)

        # Check the exception and emit pynisher specific ones too
        task.on_exception(self._check_to_emit_pynisher_exception)

    def _check_to_emit_pynisher_exception(self, exception: BaseException) -> None:
        """Check if the exception is a pynisher exception and emit it."""
        if isinstance(exception, Pynisher.CpuTimeoutException):
            self.task.emit_many(
                {
                    self.TIMEOUT: ((exception,), None),
                    self.CPU_TIME_LIMIT_REACHED: ((exception,), None),
                },
            )
        elif isinstance(exception, self.WallTimeoutException):
            self.task.emit_many(
                {
                    self.TIMEOUT: ((exception,), None),
                    self.WALL_TIME_LIMIT_REACHED: ((exception,), None),
                },
            )
        elif isinstance(exception, self.MemoryLimitException):
            self.task.emit(self.MEMORY_LIMIT_REACHED, exception)
