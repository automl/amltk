"""The
[`WarningFilter`][amltk.scheduling.plugins.warning_filter.WarningFilter]
if used to automatically filter out warnings from a [`Task`][amltk.scheduling.task.Task]
as it runs.

This wraps your function in context manager
[`warnings.catch_warnings()`][warnings.catch_warnings]
and applies your arguments to [`warnings.filterwarnings()`][warnings.filterwarnings],
as you would normally filter warnings in Python.

??? tip "Usage"

    ```python exec="true" source="material-block" html="true"
    import warnings
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import WarningFilter

    def f() -> None:
        warnings.warn("This is a warning")

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(f, plugins=WarningFilter("ignore"))
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar
from typing_extensions import ParamSpec, Self, override

from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")
TrialInfo = TypeVar("TrialInfo")


class _IgnoreWarningWrapper(Generic[P, R]):
    """A wrapper to ignore warnings."""

    def __init__(
        self,
        fn: Callable[P, R],
        *warning_args: Any,
        **warning_kwargs: Any,
    ):
        """Initialize the wrapper.

        Args:
            fn: The function to wrap.
            *warning_args: arguments to pass to
                [`warnings.filterwarnings()`][warnings.filterwarnings].
            **warning_kwargs: keyword arguments to pass to
                [`warnings.filterwarnings()`][warnings.filterwarnings].
        """
        super().__init__()
        self.fn = fn
        self.warning_args = warning_args
        self.warning_kwargs = warning_kwargs

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        with warnings.catch_warnings():
            warnings.filterwarnings(*self.warning_args, **self.warning_kwargs)
            return self.fn(*args, **kwargs)


class WarningFilter(Plugin):
    """A plugin that disables warnings emitted from tasks."""

    name: ClassVar = "warning-filter"
    """The name of the plugin."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the plugin.

        Args:
            *args: arguments to pass to
                [`warnings.filterwarnings`][warnings.filterwarnings].
            **kwargs: keyword arguments to pass to
                [`warnings.filterwarnings`][warnings.filterwarnings].
        """
        super().__init__()
        self.task: Task | None = None
        self.warning_args = args
        self.warning_kwargs = kwargs

    @override
    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

    @override
    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict]:
        """Pre-submit hook.

        Wraps the function to ignore warnings.
        """
        wrapped_f = _IgnoreWarningWrapper(fn, *self.warning_args, **self.warning_kwargs)
        return wrapped_f, args, kwargs

    @override
    def copy(self) -> Self:
        """Return a copy of the plugin."""
        return self.__class__(*self.warning_args, **self.warning_kwargs)

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Table

        table = Table("Args", "Kwargs", padding=(0, 1), show_edge=False, box=None)
        table.add_row(Pretty(self.warning_args), Pretty(self.warning_kwargs))
        return Panel(table, title=f"Plugin {self.name}")
