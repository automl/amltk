"""The [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
uses [pynisher](https://github.com/automl/pynisher) to place **memory**, **walltime**
and **cputime** constraints on processes, crashing them if these limits are reached.
These default units are `bytes ("B")` and `seconds ("s")` but you can also use other
units, please see the relevant API doc.

It's best use is when used with
[`Scheduler.with_processes()`][amltk.scheduling.Scheduler.with_processes] to have work
performed in processes.

!!! tip "Requirements"

    This required `pynisher` which can be installed with:

    ```bash
    pip install amltk[pynisher]

    # Or directly
    pip install pynisher
    ```

??? tip "Usage"

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Task, Scheduler
    from amltk.scheduling.plugins.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        time.sleep(x)
        return 42

    scheduler = Scheduler.with_processes()
    task = scheduler.task(f, plugins=PynisherPlugin(walltime_limit=(1, "s")))

    @task.on("pynisher-timeout")
    def callback(exception):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```

??? example "`@events`"

    === "`@pynisher-timeout`"

        ::: amltk.scheduling.plugins.pynisher.PynisherPlugin.TIMEOUT

    === "`@pynisher-memory-limit`"

        ::: amltk.scheduling.plugins.pynisher.PynisherPlugin.MEMORY_LIMIT_REACHED

    === "`@pynisher-cputime-limit`"

        ::: amltk.scheduling.plugins.pynisher.PynisherPlugin.CPU_TIME_LIMIT_REACHED

    === "`@pynisher-walltime-limit`"

        ::: amltk.scheduling.plugins.pynisher.PynisherPlugin.WALL_TIME_LIMIT_REACHED

??? warning "Scheduler Executor"

    This will place process limits on the task as soon as it starts
    running, whever it may be running. If you are using
    [`Scheduler.with_sequential()`][amltk.Scheduler.with_sequential]
    then this will place limits on the main process, likely not what you
    want. This also does not work with a
    [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor].

    If using this with something like [`dask-jobqueue`],
    then this will place limits on the workers it spawns. It would be better
    to place limits directly through dask job-queue then.

??? warning "Platform Limitations (Mac, Windows)"

    Pynisher has some limitations with memory on Mac and Windows:
    https://github.com/automl/pynisher#features

    You can check this with `PynisherPlugin.supports("memory")`,
    `PynisherPlugin.supports("cpu_time")` and
    `PynisherPlugin.supports("wall_time")`.
    See [`PynisherPlugin.supports()`][amltk.scheduling.plugins.pynisher.PynisherPlugin.supports]
"""  # noqa: E501
from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypeVar
from typing_extensions import ParamSpec, Self, override

import pynisher
import pynisher.exceptions

from amltk.optimization import Trial
from amltk.scheduling.events import Event
from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    import asyncio

    from rich.panel import Panel

    from amltk.scheduling.task import Task

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _PynisherWrap(Generic[P, R]):
    fn: Callable[P, R]
    memory_limit: int | tuple[int, str] | None = None
    cputime_limit: int | tuple[float, str] | None = None
    walltime_limit: int | tuple[float, str] | None = None
    terminate_child_processes: bool = True
    context: BaseContext | None = None
    disable_trial_handling: bool = False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if any(
            limit is not None
            for limit in (self.memory_limit, self.cputime_limit, self.walltime_limit)
        ):
            fn = pynisher.Pynisher(
                self.fn,
                memory=self.memory_limit,
                cpu_time=self.cputime_limit,
                wall_time=self.walltime_limit,
                terminate_child_processes=True,
                context=self.context,
            )
        else:
            fn = self.fn

        trial: Trial | None = None
        if not self.disable_trial_handling:
            if len(args) > 0 and isinstance(args[0], Trial):
                trial = args[0]
            elif (_trial := kwargs.get("trial")) is not None:
                if not isinstance(_trial, Trial):
                    raise ValueError(
                        f"Expected 'trial' to be a Trial instance, got {type(trial)}"
                        f"\n{trial=}",
                    )
                trial = _trial

        if trial is not None:
            try:
                return fn(*args, **kwargs)
            except pynisher.PynisherException as e:
                tb = traceback.format_exc()
                return trial.fail(e, tb)  # type: ignore
        else:
            return fn(*args, **kwargs)


class PynisherPlugin(Plugin):
    """A plugin that wraps a task in a pynisher to enforce limits on it.

    This plugin wraps a task function in a `Pynisher` instance to enforce
    limits on the task. The limits are set by any of `memory_limit=`,
    `cpu_time_limit=` and `wall_time_limit=`.

    Adds four new events to the task

    * [`@pynisher-timeout`][amltk.scheduling.plugins.pynisher.PynisherPlugin.TIMEOUT]
    * [`@pynisher-memory-limit`][amltk.scheduling.plugins.pynisher.PynisherPlugin.MEMORY_LIMIT_REACHED]
    * [`@pynisher-cputime-limit`][amltk.scheduling.plugins.pynisher.PynisherPlugin.CPU_TIME_LIMIT_REACHED]
    * [`@pynisher-walltime-limit`][amltk.scheduling.plugins.pynisher.PynisherPlugin.WALL_TIME_LIMIT_REACHED]

    Attributes:
        memory_limit: The memory limit of the task.
        cpu_time_limit: The cpu time limit of the task.
        wall_time_limit: The wall time limit of the task.
    """  # noqa: E501

    name: ClassVar = "pynisher-plugin"
    """The name of the plugin."""

    TIMEOUT: Event[PynisherPlugin.TimeoutException] = Event("pynisher-timeout")
    """A Task timed out, either due to the wall time or cpu time limit.

    Will call any subscribers with the exception as the argument.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Task, Scheduler
    from amltk.scheduling.plugins.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        time.sleep(x)
        return 42

    scheduler = Scheduler.with_processes()
    task = scheduler.task(f, plugins=PynisherPlugin(walltime_limit=(1, "s")))

    @task.on("pynisher-timeout")
    def callback(exception):
        pass
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    MEMORY_LIMIT_REACHED: Event[pynisher.exceptions.MemoryLimitException] = Event(
        "pynisher-memory-limit",
    )
    """A Task was submitted but reached it's memory limit.

    Will call any subscribers with the exception as the argument.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Task, Scheduler
    from amltk.scheduling.plugins.pynisher import PynisherPlugin
    import numpy as np

    def f(x: int) -> int:
        x = np.arange(100000000)
        time.sleep(x)
        return 42

    scheduler = Scheduler.with_processes()
    task = scheduler.task(f, plugins=PynisherPlugin(memory_limit=(1, "KB")))

    @task.on("pynisher-memory-limit")
    def callback(exception):
        pass

    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    CPU_TIME_LIMIT_REACHED: Event[pynisher.exceptions.CpuTimeoutException] = Event(
        "pynisher-cputime-limit",
    )
    """A Task was submitted but reached it's cpu time limit.

    Will call any subscribers with the exception as the argument.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Task, Scheduler
    from amltk.scheduling.plugins.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        i = 0
        while True:
            # Keep busying computing the answer to everything
            i += 1

        return 42

    scheduler = Scheduler.with_processes()
    task = scheduler.task(f, plugins=PynisherPlugin(cputime_limit=(1, "s")))

    @task.on("pynisher-cputime-limit")
    def callback(exception):
        pass

    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    WALL_TIME_LIMIT_REACHED: Event[pynisher.exceptions.WallTimeoutException] = Event(
        "pynisher-walltime-limit",
    )
    """A Task was submitted but reached it's wall time limit.

    Will call any subscribers with the exception as the argument.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Task, Scheduler
    from amltk.scheduling.plugins.pynisher import PynisherPlugin
    import time

    def f(x: int) -> int:
        time.sleep(x)
        return 42

    scheduler = Scheduler.with_processes()
    task = scheduler.task(f, plugins=PynisherPlugin(walltime_limit=(1, "s")))

    @task.on("pynisher-walltime-limit")
    def callback(exception):
        pass

    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    PynisherException: TypeAlias = pynisher.exceptions.PynisherException
    """The base exception for all pynisher exceptions."""

    TimeoutException: TypeAlias = pynisher.exceptions.TimeoutException
    """The exception that is raised when a task times out."""

    MemoryLimitException: TypeAlias = pynisher.exceptions.MemoryLimitException
    """The exception that is raised when a task reaches it's memory limit."""

    CpuTimeoutException: TypeAlias = pynisher.exceptions.CpuTimeoutException
    """The exception that is raised when a task reaches it's cpu time limit."""

    WallTimeoutException: TypeAlias = pynisher.exceptions.WallTimeoutException
    """The exception that is raised when a task reaches it's wall time limit."""

    def __init__(
        self,
        *,
        memory_limit: int | tuple[int, str] | None = None,
        cputime_limit: int | tuple[float, str] | None = None,
        walltime_limit: int | tuple[float, str] | None = None,
        context: BaseContext | None = None,
        disable_trial_handling: bool = False,
    ):
        """Initialize a `PynisherPlugin` instance.

        Args:
            memory_limit: The memory limit to wrap the task in. Base unit is in bytes
                but you can specify `(value, unit)` where `unit` is one of
                `("B", "KB", "MB", "GB")`. Defaults to `None`
            cputime_limit: The cpu time limit to wrap the task in. Base unit is in
                seconds but you can specify `(value, unit)` where `unit` is one of
                `("s", "m", "h")`. Defaults to `None`
            walltime_limit: The wall time limit for the task. Base unit is in seconds
                but you can specify `(value, unit)` where `unit` is one of
                `("s", "m", "h")`. Defaults to `None`.
            context: The context to use for multiprocessing. Defaults to `None`.
                See [`multiprocessing.get_context()`][multiprocessing.get_context]
            disable_trial_handling: By default, the `PynisherPlugin` will auto-detect
                if the task is one for a `Trial`. If so, it will catch any pynisher
                specific exceptions and return a `Trial.Report` with `trial.fail()`,
                instead of raising the expcetion. This has the effect that the report
                can be caught with `task.on_result` where it can be handled. This will
                also prevent the specific events `@pynisher-timeout`,
                `@pynisher-memory-limit`, `@pynisher-cputime-limit`
                and `@pynisher-walltime-limit` from being emitted.

                If this
                is `True`, then the pynisher exceptions will be raised as normal and
                should be handled with `task.on_exception` where there is no direct
                access to the `Trial` submitted.

                ??? note "Auto-Detection"

                    This will be triggered if the first positional argument is a
                    `Trial` or if any of the keyword arguments are `"trial"`.

                    ```python
                    from amltk.optimization import Trial
                    from amltk.scheduling import Scheduler

                    def trial_evaluator_one(trial: Trial, ...) -> int:
                        ...

                    def trial_evaluator_two(..., trial: Trial) -> int:
                        ...

                    scheduler = Scheduler.with_processes(1)

                    task_one = scheduler.task(
                        trial_evaluator_one,
                        plugins=PynisherPlugin(memory_limit=(1, "gb")
                    )
                    task_two = scheduler.task(
                        trial_evaluator_two,
                        plugins=PynisherPlugin(memory_limit=(1, "gb")
                    )

                    # Will auto-detect
                    trial = Trial(...)
                    task_one.submit(trial, ...)
                    task_two.submit(..., trial=trial)

                    # Will not auto-detect
                    task_one.submit(42, trial)
                    ```
        """
        super().__init__()

        for limit, name in [
            (memory_limit, "memory"),
            (cputime_limit, "cpu_time"),
            (walltime_limit, "wall_time"),
        ]:
            if limit is not None and not self.supports(name):  # type: ignore
                raise RuntimeError(
                    f"Your platform does not support {name} limits."
                    " Please see pynisher documentation for more:"
                    "\nhttps://github.com/automl/pynisher#features",
                )

        self.memory_limit = memory_limit
        self.cputime_limit = cputime_limit
        self.walltime_limit = walltime_limit
        self.context = context
        self.disable_trial_handling = disable_trial_handling

        self.task: Task

    @override
    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[Callable[P, R], tuple, dict]:
        """Wrap a task function in a `Pynisher` instance."""
        _fn = _PynisherWrap(
            fn,
            disable_trial_handling=self.disable_trial_handling,
            memory_limit=self.memory_limit,
            cputime_limit=self.cputime_limit,
            walltime_limit=self.walltime_limit,
            context=self.context,
            terminate_child_processes=True,
        )
        return _fn, args, kwargs

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

        Please see [`Plugin.copy()`][amltk.Plugin.copy].
        """
        return self.__class__(
            memory_limit=self.memory_limit,
            cputime_limit=self.cputime_limit,
            walltime_limit=self.walltime_limit,
        )

    def _check_to_emit_pynisher_exception(
        self,
        _: asyncio.Future,
        exception: BaseException,
    ) -> None:
        """Check if the exception is a pynisher exception and emit it."""
        if isinstance(exception, pynisher.CpuTimeoutException):
            self.task.emitter.emit(self.TIMEOUT, exception)
            self.task.emitter.emit(self.CPU_TIME_LIMIT_REACHED, exception)
        elif isinstance(exception, pynisher.WallTimeoutException):
            self.task.emitter.emit(self.TIMEOUT)
            self.task.emitter.emit(self.WALL_TIME_LIMIT_REACHED, exception)
        elif isinstance(exception, pynisher.MemoryLimitException):
            self.task.emitter.emit(self.MEMORY_LIMIT_REACHED, exception)

    @classmethod
    def supports(cls, kind: Literal["wall_time", "cpu_time", "memory"]) -> bool:
        """Check if the task is supported by the plugin.

        Args:
            kind: The kind of limit to check.

        Returns:
            `True` if the limit is supported by the plugin for your os, else `False`.
        """
        return pynisher.supports(kind)

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Table

        table = Table(
            "Memory",
            "Wall Time",
            "CPU Time",
            padding=(0, 1),
            show_edge=False,
            box=None,
        )
        table.add_row(
            Pretty(self.memory_limit),
            Pretty(self.walltime_limit),
            Pretty(self.cputime_limit),
        )
        return Panel(table, title=f"Plugin {self.name}")
