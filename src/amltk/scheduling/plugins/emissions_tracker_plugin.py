from codecarbon import EmissionsTracker
from typing import Callable, Any, ClassVar, TypeVar
from typing_extensions import Self

from amltk.scheduling.plugins.plugin import Plugin
from amltk.scheduling.task import Task

P = TypeVar("P")
R = TypeVar("R")


class _EmissionsTrackerWrapper:
    """A wrapper around codecarbon package to measure emissions."""

    def __init__(
            self,
            fn: Callable[[P], R],
            task: Task,
            *codecarbon_args: Any,
            **codecarbon_kwargs: Any,
    ):
        """Initialize the wrapper.

        Args:
            fn: The function to wrap.
            task: The task the function is being wrapped for.
            *codecarbon_args: arguments to pass to
                [`codecarbon.EmissionsTracker`][codecarbon.EmissionsTracker].
            **codecarbon_kwargs: keyword arguments to pass to
                [`codecarbon.EmissionsTracker`][codecarbon.EmissionsTracker].
        """
        super().__init__()
        self.fn = fn
        self.task = task
        self.codecarbon_args = codecarbon_args
        self.codecarbon_kwargs = codecarbon_kwargs

    def __call__(self, *args: any, **kwargs: any) -> R:
        with EmissionsTracker(*self.codecarbon_args, **self.codecarbon_kwargs) as tracker:
            result = self.fn(*args, **kwargs)
            return result


class EmissionsTrackerPlugin(Plugin):
    """A plugin that tracks carbon emissions using codecarbon library."""

    name: ClassVar = "emissions-tracker"
    """The name of the plugin."""

    """
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
        EmissionsTrackerPlugin(log_level="info", save_to_file=False) # pass any codecarbon args here
    ])

    @scheduler.on_start
    def on_start():
        task.submit(5) # submit any args here

    @task.on_submitted
    def on_submitted(future, *args, **kwargs):
        print(f"Task was submitted", future, args, kwargs) 

    @task.on_done
    def on_done(future):
        print("Task done: ", future.result()) # result is the return value of the function

    scheduler.run()    
    ```  
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.task: Task | None = None
        self.codecarbon_args = args
        self.codecarbon_kwargs = kwargs

    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

    def pre_submit(
            self,
            fn: Callable[[P], R],
            *args: any,
            **kwargs: any,
    ) -> tuple[Callable[[P], R], tuple, dict]:
        """Pre-submit hook."""
        wrapped_f = _EmissionsTrackerWrapper(fn, self.task, *self.codecarbon_args, **self.codecarbon_kwargs)
        return wrapped_f, args, kwargs

    def copy(self) -> Self:
        """Return a copy of the plugin."""
        return self.__class__(*self.codecarbon_args, **self.codecarbon_kwargs)

    def __rich__(self):
        """Return a rich panel."""
        from rich.panel import Panel

        return Panel(
            f"codecarbon_args: {self.codecarbon_args} codecarbon_kwargs: {self.codecarbon_kwargs}",
            title=f"Plugin {self.name}"
        )
