"""The [`Limiter`][amltk.scheduling.plugins.Limiter] can limit the number of
times a function is called, how many concurrent instances of it can be running,
or whether it can run while another task is running.

The functionality of the `Limiter` could also be implemented without a plugin but
it gives some nice utility.

??? tip "Usage"

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Limiter

    def fn(x: int) -> int:
        return x + 1

    scheduler = Scheduler.with_processes(1)

    task = scheduler.task(fn, plugins=[Limiter(max_calls=2)])

    @task.on("call-limit-reached")
    def callback(task: Task, *args, **kwargs):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```

??? example "`@events`"

    === "`@call-limit-reached`"

        ::: amltk.scheduling.plugins.Limiter.CALL_LIMIT_REACHED

    === "`@concurrent-limit-reached`"

        ::: amltk.scheduling.plugins.Limiter.CONCURRENT_LIMIT_REACHED

    === "`@disabled-due-to-running-task`"

        ::: amltk.scheduling.plugins.Limiter.DISABLED_DUE_TO_RUNNING_TASK
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from typing_extensions import ParamSpec, Self, override

from amltk.scheduling.events import Event
from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")
TrialInfo = TypeVar("TrialInfo")


class Limiter(Plugin):
    """A plugin that limits the submission of a task.

    Adds three new events to the task:

    * [`@call-limit-reached`][amltk.scheduling.plugins.Limiter.CALL_LIMIT_REACHED]
    * [`@concurrent-limit-reached`][amltk.scheduling.plugins.Limiter.CONCURRENT_LIMIT_REACHED]
    * [`@disabled-due-to-running-task`][amltk.scheduling.plugins.Limiter.DISABLED_DUE_TO_RUNNING_TASK]
    """  # noqa: E501

    name: ClassVar = "limiter"
    """The name of the plugin."""

    CALL_LIMIT_REACHED: Event[...] = Event("call-limit-reached")
    """The event emitted when the task has reached its call limit.

    Will call any subscribers with the task as the first argument,
    followed by the arguments and keyword arguments that were passed to the task.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Limiter

    def fn(x: int) -> int:
        return x + 1

    scheduler = Scheduler.with_processes(1)

    task = scheduler.task(fn, plugins=[Limiter(max_calls=2)])

    @task.on("call-limit-reached")
    def callback(task: Task, *args, **kwargs):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    CONCURRENT_LIMIT_REACHED: Event[...] = Event("concurrent-limit-reached")
    """The event emitted when the task has reached its concurrent call limit.

    Will call any subscribers with the task as the first argument, followed by the
    arguments and keyword arguments that were passed to the task.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Limiter

    def fn(x: int) -> int:
        return x + 1

    scheduler = Scheduler.with_processes(2)

    task = scheduler.task(fn, plugins=[Limiter(max_concurrent=2)])

    @task.on("concurrent-limit-reached")
    def callback(task: Task, *args, **kwargs):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    DISABLED_DUE_TO_RUNNING_TASK: Event[...] = Event("disabled-due-to-running-task")
    """The event emitter when the task was not submitted due to some other
    running task.

    Will call any subscribers with the task as first argument, followed by
    the arguments and keyword arguments that were passed to the task.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Limiter

    def fn(x: int) -> int:
        return x + 1

    scheduler = Scheduler.with_processes(2)

    other_task = scheduler.task(fn)
    task = scheduler.task(fn, plugins=[Limiter(not_while_running=other_task)])

    @task.on("disabled-due-to-running-task")
    def callback(task: Task, *args, **kwargs):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        max_concurrent: int | None = None,
        not_while_running: Task | Iterable[Task] | None = None,
    ):
        """Initialize the plugin.

        Args:
            max_calls: The maximum number of calls to the task.
            max_concurrent: The maximum number of calls of this task that can
                be in the queue.
            not_while_running: A task or iterable of tasks that if active, will prevent
                this task from being submitted.
        """
        super().__init__()

        if not_while_running is None:
            not_while_running = []
        elif isinstance(not_while_running, Iterable):
            not_while_running = list(not_while_running)
        else:
            not_while_running = [not_while_running]

        self.max_calls = max_calls
        self.max_concurrent = max_concurrent
        self.not_while_running = not_while_running
        self.task: Task | None = None

        if isinstance(max_calls, int) and not max_calls > 0:
            raise ValueError("max_calls must be greater than 0")

        if isinstance(max_concurrent, int) and not max_concurrent > 0:
            raise ValueError("max_concurrent must be greater than 0")

        self._calls = 0

    @override
    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

        if self.task in self.not_while_running:
            raise ValueError(
                f"Task {self.task} was found in the {self.not_while_running=}"
                " list. This is disabled but please raise an issue if you think this"
                " has sufficient use case.",
            )

        task.emitter.add_event(
            self.CALL_LIMIT_REACHED,
            self.CONCURRENT_LIMIT_REACHED,
            self.DISABLED_DUE_TO_RUNNING_TASK,
        )

        # Make sure to increment the count when a task was submitted
        task.on_submitted(self._increment_call_count, hidden=True)

    @property
    def n_running(self) -> int:
        """Return the number of running tasks."""
        assert self.task is not None
        return len(self.task.queue)

    @override
    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict] | None:
        """Pre-submit hook.

        Prevents submission of the task if it exceeds any of the set limits.
        """
        assert self.task is not None

        if self.max_calls is not None and self._calls >= self.max_calls:
            self.task.emitter.emit(self.CALL_LIMIT_REACHED, self.task, *args, **kwargs)
            return None

        if self.max_concurrent is not None and self.n_running >= self.max_concurrent:
            self.task.emitter.emit(
                self.CONCURRENT_LIMIT_REACHED,
                self.task,
                *args,
                **kwargs,
            )
            return None

        for other_task in self.not_while_running:
            if other_task.running():
                self.task.emitter.emit(
                    self.DISABLED_DUE_TO_RUNNING_TASK,
                    other_task,
                    self.task,
                    *args,
                    **kwargs,
                )
                return None

        return fn, args, kwargs

    @override
    def copy(self) -> Self:
        """Return a copy of the plugin."""
        return self.__class__(
            max_calls=self.max_calls,
            max_concurrent=self.max_concurrent,
        )

    def _increment_call_count(self, *_: Any, **__: Any) -> None:
        self._calls += 1

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        from amltk._richutil import Function

        table = Table.grid(padding=(0, 1))
        if self.max_calls is not None:
            table.add_row("Calls", f"{self._calls}/{self.max_calls}")

        if self.max_concurrent is not None:
            table.add_row("Concurrent", f"{self.n_running}/{self.max_concurrent}")

        for task in self.not_while_running:
            f = Function(
                task.function,
                signature="...",
                link=False,
            )
            if task.running():
                table.add_row(
                    "Not While",
                    f,
                    Text(task.unique_ref, "italic, yellow"),
                    Text("Running", style="bold green"),
                )
            else:
                table.add_row(
                    "Not While",
                    f,
                    Text("Ref: ").append(task.unique_ref, "italic yellow"),
                )

        return Panel(table, title=f"Plugin {self.name}")
