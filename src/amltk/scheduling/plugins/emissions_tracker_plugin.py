from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from typing import Callable, Generic, TypeVar
from typing_extensions import ParamSpec, override

from codecarbon import EmissionsTracker
from amltk.scheduling.plugins.plugin import Plugin
from amltk.scheduling.events import Event

if TYPE_CHECKING:
    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")


class _EmissionsTrackerWrapper(Generic[P, R]):
    """A wrapper around codecarbon package to measure emissions."""

    def __init__(
            self,
            fn: Callable[P, R],
            task: Task,
            *codecarbon_args: Any,
            **codecarbon__kwargs: Any,
    ):
        """Initialize the wrapper.

        Args:
            fn: The function to wrap.
        """
        super().__init__()
        self.fn = fn
        self.task = task
        self.codecarbon_args = codecarbon_args
        self.codecarbon__kwargs = codecarbon__kwargs

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        with EmissionsTracker(*self.codecarbon_args, **self.codecarbon__kwargs) as tracker:
            result = self.fn(*args, **kwargs)
            emissions_data = tracker.get_emissions()
            self.task.emitter.emit(EmissionsTrackerPlugin.EMISSIONS_TRACKED, emissions_data)
            return result


class EmissionsTrackerPlugin(Plugin):
    """A plugin that tracks carbon emissions using codecarbon library."""

    name: ClassVar = "emissions-tracker"
    """The name of the plugin."""

    EMISSIONS_TRACKED: Event[...] = Event("emissions-tracked")
    """The event emitted when emissions are tracked.

    Will call any subscribers with the task as the first argument,
    followed by the emissions data.

    ```python
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import EmissionsTrackerPlugin

    def fn(x: int) -> int:
        return x + 1

    scheduler = Scheduler.with_processes(1)

    # Add the EmissionsTrackerPlugin to the list of plugins
    task = scheduler.task(fn, plugins=[EmissionsTrackerPlugin()])

    @task.on("emissions-tracked")
    def callback(task: Task, emissions_data: dict):
        # Handle emissions data
        pass
    ```
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.task: Task | None = None
        self.codecarbon_args = args
        self.codecarbon_kwargs = kwargs

    @override
    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task
        task.emitter.add_event(self.EMISSIONS_TRACKED)

    @override
    def pre_submit(
            self,
            fn: Callable[P, R],
            *args: any,
            **kwargs: any,
    ) -> tuple[Callable[P, R], tuple, dict]:
        """Pre-submit hook."""
        wrapped_f = _EmissionsTrackerWrapper(fn, self.task, *self.codecarbon_args, **self.codecarbon_kwargs)
        return wrapped_f, args, kwargs

    @override
    def copy(self) -> EmissionsTrackerPlugin:
        """Return a copy of the plugin."""
        return self.__class__(*self.codecarbon_args, **self.codecarbon_kwargs)

    @override
    def __rich__(self):
        """Return a rich panel."""
        from rich.panel import Panel

        return Panel(
            f"codecarbon_args: {self.codecarbon_args} codecarbon_kwargs: {self.codecarbon_kwargs}",
            title=f"Plugin {self.name}"
        )
