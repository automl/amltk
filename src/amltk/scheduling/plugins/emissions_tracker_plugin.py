"""Emissions Tracker Plugin Module.

This module defines a plugin for tracking carbon emissions using the codecarbon library.

For usage examples, refer to the docstring of the EmissionsTrackerPlugin class.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar
from typing_extensions import ParamSpec, Self

from codecarbon import EmissionsTracker

from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")


class _EmissionsTrackerWrapper(Generic[P, R]):
    """A wrapper around codecarbon package to measure emissions."""

    def __init__(
        self,
        fn: Callable[P, R],
        *codecarbon_args: Any,
        **codecarbon_kwargs: Any,
    ):
        """Initialize the wrapper.

        Args:
            fn: The function to wrap.
            *codecarbon_args: Arguments to pass to codecarbon EmissionsTracker.
            **codecarbon_kwargs: Keyword args to pass to codecarbon EmissionsTracker.
        """
        super().__init__()
        self.fn = fn
        self.codecarbon_args = codecarbon_args
        self.codecarbon_kwargs = codecarbon_kwargs

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        with EmissionsTracker(*self.codecarbon_args, **self.codecarbon_kwargs):
            return self.fn(*args, **kwargs)


class EmissionsTrackerPlugin(Plugin):
    """A plugin that tracks carbon emissions using codecarbon library.

    Usage Example:

    ```python
    from concurrent.futures import ThreadPoolExecutor
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins.emissions_tracker_plugin import EmissionsTrackerPlugin

    def some_function(x: int) -> int:
        return x * 2

    executor = ThreadPoolExecutor(max_workers=1)

    # Create a Scheduler instance with the executor
    scheduler = Scheduler(executor=executor)

    # Create a task with the emissions tracker plugin
    task = scheduler.task(some_function, plugins=[
        # Pass any codecarbon parameters as args here
        EmissionsTrackerPlugin(log_level="info", save_to_file=False)
    ])

    @scheduler.on_start
    def on_start():
        task.submit(5)  # Submit any args here

    @task.on_submitted
    def on_submitted(future, *args, **kwargs):
        print(f"Task was submitted", future, args, kwargs)

    @task.on_done
    def on_done(future):
        # Result is the return value of the function
        print("Task done: ", future.result())

    scheduler.run()
    ```
    """

    name: ClassVar = "emissions-tracker"
    """The name of the plugin."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the EmissionsTrackerPlugin.

        Args:
            *args: Additional arguments to pass to codecarbon library.
            **kwargs: Additional keyword arguments to pass to codecarbon library.

        You can pass any codecarbon parameters as args to EmissionsTrackerPlugin.
        Please refer to the official codecarbon documentation for more details:
        https://mlco2.github.io/codecarbon/parameters.html
        """
        super().__init__()
        self.task: Task | None = None
        self.codecarbon_args = args
        self.codecarbon_kwargs = kwargs

    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict]:
        """Pre-submit hook."""
        wrapped_f = _EmissionsTrackerWrapper(
            fn,
            self.task,
            *self.codecarbon_args,
            **self.codecarbon_kwargs,
        )
        return wrapped_f, args, kwargs

    def copy(self) -> Self:
        """Return a copy of the plugin."""
        return self.__class__(*self.codecarbon_args, **self.codecarbon_kwargs)

    def __rich__(self) -> Panel:
        """Return a rich panel."""
        from rich.panel import Panel

        return Panel(
            f"codecarbon_args: {self.codecarbon_args} "
            f"codecarbon_kwargs: {self.codecarbon_kwargs}",
            title=f"Plugin {self.name}",
        )
