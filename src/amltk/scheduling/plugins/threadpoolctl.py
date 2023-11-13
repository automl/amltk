"""The
[`ThreadPoolCTLPlugin`][amltk.scheduling.plugins.threadpoolctl.ThreadPoolCTLPlugin]
if useful for parallel training of models. Without limiting with
threadpoolctl, the number of threads used by a given model may
oversubscribe to resources and cause significant slowdowns.

This is the mechanism employed by scikit-learn to limit the number of
threads used by a given model.

See [threadpoolctl documentation](https://github.com/joblib/threadpoolctl).

!!! tip "Requirements"

    This requires `threadpoolctl` which can be installed with:

    ```bash
    pip install amltk[threadpoolctl]

    # Or directly
    pip install threadpoolctl
    ```

??? tip "Usage"

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins.threadpoolctl import ThreadPoolCTLPlugin

    scheduler = Scheduler.with_processes(1)

    def f() -> None:
        # ... some task that respects the limits set by threadpoolctl
        pass

    task = scheduler.task(f, plugins=ThreadPoolCTLPlugin(max_threads=1))
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar
from typing_extensions import ParamSpec, Self, override

from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class _ThreadPoolLimiter(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        max_threads: int | dict[str, int] | None = None,
        user_api: str | None = None,
    ):
        super().__init__()
        self.fn = fn
        self.max_threads = max_threads
        self.user_api = user_api

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        import threadpoolctl

        controller = threadpoolctl.ThreadpoolController()
        with controller.limit(limits=self.max_threads, user_api=self.user_api):
            logger.debug(f"threadpoolctl: {controller.info()}")
            return self.fn(*args, **kwargs)


class ThreadPoolCTLPlugin(Plugin):
    """A plugin that limits the usage of threads in a task.

    This plugin is used to make utilize threadpoolctl with tasks,
    useful for parallel training of models. Without limiting with
    threadpoolctl, the number of threads used by a given model may
    oversubscribe to resources and cause significant slowdowns.

    Attributes:
        max_calls: The maximum number of calls to the task.
        max_concurrent: The maximum number of calls of this task that can
            be in the queue.
    """

    name: ClassVar = "threadpoolctl-plugin"
    """The name of the plugin."""

    @override
    def __init__(
        self,
        max_threads: int | dict[str, int] | None = None,
        user_api: str | None = None,
    ):
        """Initialize the plugin.

        Args:
            max_threads: The maximum number of threads to use.
            user_api: The user API to limit.
        """
        super().__init__()
        self.max_threads = max_threads
        self.user_api = user_api
        self.task: Task | None = None

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
    ) -> tuple[Callable[P, R], tuple, dict] | None:
        """Pre-submit hook.

        Wrap the function in something that will activate threadpoolctl
        when the function is called.
        """
        assert self.task is not None

        fn = _ThreadPoolLimiter(
            fn=fn,
            max_threads=self.max_threads,
            user_api=self.user_api,
        )
        return fn, args, kwargs

    @override
    def copy(self) -> Self:
        """Return a copy of the plugin.

        Please see [`Plugin.copy()`][amltk.Plugin.copy].
        """
        return self.__class__(max_threads=self.max_threads, user_api=self.user_api)

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Table

        table = Table(
            "Max Threads",
            "User-API",
            padding=(0, 1),
            show_edge=False,
            box=None,
        )
        table.add_row(Pretty(self.max_threads), Pretty(self.user_api))
        return Panel(table, title=f"Plugin {self.name}")
