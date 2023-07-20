"""Plugin for threadpoolctl.

This plugin is used to make utilize threadpoolctl with tasks,
useful for parallel training of models. Without limiting with
threadpoolctl, the number of threads used by a given model may
oversubscribe to resources and cause significant slowdowns.

See [threadpoolctl documentation](https://github.com/joblib/threadpoolctl).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar
from typing_extensions import ParamSpec, Self

from amltk.scheduling.task_plugin import TaskPlugin

if TYPE_CHECKING:
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
        self.fn = fn
        self.max_threads = max_threads
        self.user_api = user_api

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        import threadpoolctl

        controller = threadpoolctl.ThreadpoolController()
        with controller.limit(limits=self.max_threads, user_api=self.user_api):
            logger.debug(f"threadpoolctl: {controller.info()}")
            return self.fn(*args, **kwargs)


class ThreadPoolCTLPlugin(TaskPlugin):
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

    def __init__(
        self,
        max_threads: int | dict[str, int] | None = None,
        user_api: str | None = None,
    ):
        """Initialize the plugin.

        See [threadpoolctl documentation](https://github.com/joblib/threadpoolctl).

        Args:
            max_threads: The maximum number of threads to use.
            user_api: The user API to limit.
        """
        self.max_threads = max_threads
        self.user_api = user_api
        self.task: Task | None = None

    def attach_task(self, task: Task) -> None:
        """Attach the plugin to a task."""
        self.task = task

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

    def copy(self) -> Self:
        """Return a copy of the plugin.

        Please see [`TaskPlugin.copy()`][amltk.TaskPlugin.copy].
        """
        return self.__class__(max_threads=self.max_threads, user_api=self.user_api)
