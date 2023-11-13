"""Wandb plugin.

!!! todo

    This plugin is experimental and out of date.

"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)
from typing_extensions import ParamSpec, Self, override

import numpy as np
import wandb
from wandb.sdk import wandb_run
from wandb.sdk.lib import RunDisabled

from amltk.scheduling.executors import SequentialExecutor
from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from amltk.optimization import Trial
    from amltk.scheduling import Scheduler, Task

P = ParamSpec("P")
R = TypeVar("R")


logger = logging.getLogger(__name__)

WRun: TypeAlias = wandb_run.Run | RunDisabled


@dataclass
class WandbParams:
    """Parameters for initializing a wandb run.

    This class is a dataclass that contains all the parameters that are used
    to initialize a wandb run. It is used by the
    [`WandbPlugin`][amltk.scheduling.plugins.wandb.WandbPlugin] to initialize a run.
    It can be modified using the
    [`modify()`][amltk.scheduling.plugins.wandb.WandbParams.modify] method.

    Please refer to the documentation of the
    [`wandb.init()`](https://docs.wandb.ai/ref/python/init) method for more information
    on the parameters.
    """

    project: str | None = None
    group: str | None = None
    job_type: str | None = None
    entity: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    reinit: bool | None = None
    config_exclude_keys: list[str] | None = None
    config_include_keys: list[str] | None = None
    resume: bool | str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    allow_val_change: bool = False
    force: bool = False
    dir: str | Path | None = None

    # TODO: There is a parameter for `id`, this should be used in multifidelity
    # runs.

    def modify(self, **kwargs: Any) -> WandbParams:
        """Modify the parameters of this instance.

        This method returns a new instance of this class with the parameters
        modified. This is useful for example when you want to modify the
        parameters of a run to add tags or notes.
        """
        return replace(self, **kwargs)

    def run(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> WRun:
        """Initialize a wandb run.

        This method initializes a wandb run using the parameters of this
        instance. It returns the wandb run object.

        Args:
            name: The name of the run.
            config: The configuration of the run.

        Returns:
            The wandb run object.
        """
        run = wandb.init(
            config=dict(config) if config else None,
            name=name,
            project=self.project,
            group=self.group,
            tags=self.tags,
            entity=self.entity,
            notes=self.notes,
            reinit=self.reinit,
            dir=self.dir,
            config_exclude_keys=self.config_exclude_keys,
            config_include_keys=self.config_include_keys,
            mode=self.mode,
            allow_val_change=self.allow_val_change,
            force=self.force,
        )
        if run is None:
            raise RuntimeError("Wandb run was not initialized")

        return run


class WandbLiveRunWrap(Generic[P]):
    """Wrap a function to log the results to a wandb run.

    This class is used to wrap a function that returns a report to log the
    results to a wandb run. It is used by the
    [`WandbTrialTracker`][amltk.scheduling.plugins.wandb.WandbTrialTracker] to wrap
    the target function.
    """

    def __init__(
        self,
        params: WandbParams,
        fn: Callable[Concatenate[Trial, P], Trial.Report],
        *,
        modify: Callable[[Trial, WandbParams], WandbParams] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            params: The parameters to initialize the wandb run.
            fn: The function to wrap.
            modify: A function that modifies the parameters of the wandb run
                before each trial.
        """
        super().__init__()
        self.params = params
        self.fn = fn
        self.modify = modify

    def __call__(self, trial: Trial, *args: P.args, **kwargs: P.kwargs) -> Trial.Report:
        """Call the wrapped function and log the results to a wandb run."""
        params = self.params if self.modify is None else self.modify(trial, self.params)
        with params.run(name=trial.name, config=trial.config) as run:
            # Make sure the run is available from the trial
            trial.extras["wandb"] = run

            report = self.fn(trial, *args, **kwargs)

            report_df = report.df()
            run.log({"table": wandb.Table(dataframe=report_df)})
            wandb_summary = {
                k: v
                for k, v in report.summary.items()
                if isinstance(v, int | float | np.number)
            }
            run.summary.update(wandb_summary)

        wandb.finish()
        return report


class WandbTrialTracker(Plugin):
    """Track trials using wandb.

    This class is a task plugin that tracks trials using wandb.
    """

    name: ClassVar[str] = "wandb-trial-tracker"
    """The name of the plugin."""

    def __init__(
        self,
        params: WandbParams,
        *,
        modify: Callable[[Trial, WandbParams], WandbParams] | None = None,
    ):
        """Initialize the plugin.

        Args:
            params: The parameters to initialize the wandb run.
            modify: A function that modifies the parameters of the wandb run
                before each trial.
        """
        super().__init__()
        self.params = params
        self.modify = modify

    @override
    def attach_task(self, task: Task) -> None:
        """Use the task to register several callbacks."""
        self._check_explicit_reinit_arg_with_executor(task.scheduler)

    @override
    def pre_submit(
        self,
        fn: Callable[P, R],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Callable[P, R], tuple, dict] | None:
        """Wrap the target function to log the results to a wandb run.

        This method wraps the target function to log the results to a wandb run
        and returns the wrapped function.

        Args:
            fn: The target function.
            args: The positional arguments of the target function.
            kwargs: The keyword arguments of the target function.

        Returns:
            The wrapped function, the positional arguments and the keyword
            arguments.
        """
        fn = WandbLiveRunWrap(self.params, fn, modify=self.modify)  # type: ignore
        return fn, args, kwargs

    @override
    def copy(self) -> Self:
        """Copy the plugin."""
        return self.__class__(modify=self.modify, params=replace(self.params))

    def _check_explicit_reinit_arg_with_executor(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Check if reinit arg was explicitly set and conflicts with executor.

        Args:
            scheduler: The scheduler to check.
        """
        if isinstance(scheduler.executor, SequentialExecutor | ThreadPoolExecutor):
            if self.params.reinit is False:
                raise ValueError(
                    "WandbPlugin reinit argument is not compatible with"
                    " SequentialExecutor or ThreadPoolExecutor. This is because"
                    " wandb.init(reinit=False) will only work if each trial is"
                    " run in a separate process. If you want to use `reinit=False`,"
                    " please consider using a different executor. Otherwise, you can"
                    " also set `reinit=True` to explicitly allow reinitialization."
                    " By default `reinit` is `None`, which means that it will be"
                    " set to `True` if the executor is not a SequentialExecutor or"
                    " ThreadPoolExecutor, and `False` otherwise.",
                )

            self.params.reinit = True


class WandbPlugin:
    """Log trials using wandb.

    This class is the entry point to log trials using wandb. It
    can be used to create a
    [`trial_tracker()`][amltk.scheduling.plugins.wandb.WandbPlugin.trial_tracker]
    to pass into a [`Task(plugins=...)`][amltk.Task] or to
    create `wandb.Run`'s for custom purposes with
    [`run()`][amltk.scheduling.plugins.wandb.WandbPlugin.run].
    """

    def __init__(
        self,
        *,
        project: str,
        group: str | None = None,
        entity: str | None = None,
        dir: str | Path | None = None,  # noqa: A002
        mode: Literal["online", "offline", "disabled"] = "online",
    ):
        """Initialize the plugin.

        Args:
            project: The name of the project.
            group: The name of the group.
            entity: The name of the entity.
            dir: The directory to store the runs in.
            mode: The mode to use for the runs.
        """
        super().__init__()
        _dir = Path(project) if dir is None else Path(dir)
        _dir.mkdir(parents=True, exist_ok=True)

        self.dir = _dir.resolve().absolute()
        self.project = project
        self.group = group
        self.entity = entity
        self.mode = mode

    def trial_tracker(
        self,
        job_type: str = "trial",
        *,
        modify: Callable[[Trial, WandbParams], WandbParams] | None = None,
    ) -> WandbTrialTracker:
        """Create a live tracker.

        Args:
            job_type: The job type to use for the runs.
            modify: A function that modifies the parameters of the wandb run
                before each trial.

        Returns:
            A live tracker.
        """
        params = WandbParams(
            project=self.project,
            entity=self.entity,
            group=self.group,
            dir=self.dir,
            mode=self.mode,  # type: ignore
            job_type=job_type,
        )
        return WandbTrialTracker(params, modify=modify)

    def run(
        self,
        *,
        name: str,
        job_type: str | None = None,
        group: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        resume: bool | str | None = None,
        notes: str | None = None,
    ) -> WRun:
        """Create a wandb run.

        See [`wandb.init()`](https://docs.wandb.ai/ref/python/init) for more.
        """
        return WandbParams(
            project=self.project,
            entity=self.entity,
            group=group,
            dir=self.dir,
            mode=self.mode,  # type: ignore
            job_type=job_type,
            tags=tags,
            resume=resume,
            notes=notes,
        ).run(
            name=name,
            config=config,
        )
