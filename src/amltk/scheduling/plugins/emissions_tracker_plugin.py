from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from typing import Callable, List, Optional

from codecarbon import EmissionsTracker
from amltk.scheduling.plugins.plugin import Plugin
from amltk.scheduling.events import Event

if TYPE_CHECKING:
    from amltk.scheduling.task import Task


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

    def __init__(self, tracker: Optional[EmissionsTracker] = None):
        super().__init__()
        self.task: Task | None = None
        self.tracker = tracker or EmissionsTracker()

    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task
        self.tracker.start()

        task.emitter.add_event(self.EMISSIONS_TRACKED)

    def pre_submit(
            self,
            fn: Callable[..., any],
            *args: any,
            **kwargs: any,
    ) -> tuple[Callable[..., any], List[any], dict[str, any]]:
        """Pre-submit hook."""
        return fn, args, kwargs

    def copy(self) -> EmissionsTrackerPlugin:
        """Return a copy of the plugin."""
        return self.__class__(tracker=self.tracker)

    def __del__(self) -> None:
        """Stop emissions tracking and perform cleanup when the plugin is deleted."""
        if self.tracker:
            data = self.tracker.stop()
            # self.task.emitter.emit(self.EMISSIONS_TRACKED, self.task, data)
