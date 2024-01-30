"""A [`Trial`][amltk.optimization.Trial] is
typically the output of
[`Optimizer.ask()`][amltk.optimization.Optimizer.ask], indicating
what the optimizer would like to evaluate next. We provide a host
of convenience methods attached to the `Trial` to make it easy to
save results, store artifacts, and more.

Paired with the `Trial` is the [`Trial.Report`][amltk.optimization.Trial.Report],
class, providing an easy way to report back to the optimizer's
[`tell()`][amltk.optimization.Optimizer.tell] with
a simple [`trial.success(cost=...)`][amltk.optimization.Trial.success] or
[`trial.fail(cost=...)`][amltk.optimization.Trial.fail] call..

### Trial

::: amltk.optimization.trial.Trial
    options:
        members: False

### Report

::: amltk.optimization.trial.Trial.Report
    options:
        members: False

"""
from __future__ import annotations

import copy
import logging
import traceback as traceback_module
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload
from typing_extensions import ParamSpec, Self, override

import numpy as np
import pandas as pd

from amltk._functional import dict_get_not_none, mapping_select, prefix_keys
from amltk._richutil.renderable import RichRenderable
from amltk._util import parse_timestamp_object
from amltk.optimization.metric import Metric, MetricCollection
from amltk.profiling import Memory, Profile, Profiler, Timer
from amltk.store import PathBucket

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.panel import Panel
    from rich.text import Text

# Inner trial info object
I = TypeVar("I")  # noqa: E741
I2 = TypeVar("I2")

# Parameters to task
P = ParamSpec("P")
P2 = ParamSpec("P2")

# Return type of task
R = TypeVar("R")

# Random TypeVar
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Trial(RichRenderable, Generic[I]):
    """A [`Trial`][amltk.optimization.Trial] encapsulates some configuration
    that needs to be evaluated. Typically, this is what is generated by an
    [`Optimizer.ask()`][amltk.optimization.Optimizer.ask] call.

    ??? tip "Usage"

        If all went smooth, your trial was successful and you can use
        [`trial.success()`][amltk.optimization.Trial.success] to generate
        a success [`Report`][amltk.optimization.Trial.Report], typically
        passing what your chosen optimizer expects, e.g., `"loss"` or `"cost"`.

        If your trial failed, you can instead use the
        [`trial.fail()`][amltk.optimization.Trial.fail] to generate a
        failure [`Report`][amltk.optimization.Trial.Report]. If use pass
        in an exception to `fail()`, it will be attached to the report along
        with any traceback it can deduce.
        Each [`Optimizer`][amltk.optimization.Optimizer] will take
        care of what to do from here.

        ```python exec="true" source="material-block" html="true"
        from amltk.optimization import Trial, Metric
        from amltk.store import PathBucket

        cost = Metric("cost", minimize=True)

        def target_function(trial: Trial) -> Trial.Report:
            x = trial.config["x"]
            y = trial.config["y"]

            with trial.profile("expensive-calculation"):
                cost = x**2 - y

            return trial.success(cost=cost)

        # ... usually obtained from an optimizer
        trial = Trial.create(
            name="some-unique-name",
            config={"x": 1, "y": 2},
            metrics=[cost]
        )

        report = target_function(trial)
        print(report.df())
        ```


    What you can return with [`trial.success()`][amltk.optimization.Trial.success]
    or [`trial.fail()`][amltk.optimization.Trial.fail] depends on the
    [`metrics`][amltk.optimization.Trial.metrics] of the trial. Typically,
    an optimizer will provide the trial with the list of metrics.

    ??? tip "Metrics"

        ::: amltk.optimization.metric.Metric
            options:
                members: False

    Some important properties are that they have a unique
    [`.name`][amltk.optimization.Trial.name] given the optimization run,
    a candidate [`.config`][amltk.optimization.Trial.config] to evaluate,
    a possible [`.seed`][amltk.optimization.Trial.seed] to use,
    and an [`.info`][amltk.optimization.Trial.info] object, which is the optimizer
    specific information, if required by you.

    !!! tip "Reporting success (or failure)"

        When using the [`success()`][amltk.optimization.trial.Trial.success]
        method, make sure to provide values for all metrics specified in the
        [`.metrics`][amltk.optimization.Trial.metrics] attribute.
        Usually these are set by the optimizer generating the `Trial`.

        If you instead report using [`fail()`][amltk.optimization.trial.Trial.success],
        any metric not specified will be set to the
        [`.worst`][amltk.optimization.Metric.worst] value of the metric.

        Each metric has a unique name, and it's crucial to use the correct names when
        reporting success, otherwise an error will occur.

        ??? example "Reporting success for metrics"

            For example:

            ```python exec="true" result="python" source="material-block"
            from amltk.optimization import Trial, Metric

            # Gotten from some optimizer usually, i.e. via `optimizer.ask()`
            trial = Trial.create(
                name="example_trial",
                config={"param": 42},
                metrics=[Metric(name="accuracy", minimize=False)]
            )

            # Incorrect usage (will raise an error)
            try:
                report = trial.success(invalid_metric=0.95)
            except ValueError as error:
                print(error)

            # Correct usage
            report = trial.success(accuracy=0.95)
            ```

    If using [`Plugins`][amltk.scheduling.plugins.Plugin], they may insert
    some extra objects in the [`.extra`][amltk.optimization.Trial.extras] dict.

    To profile your trial, you can wrap the logic you'd like to check with
    [`trial.profile()`][amltk.optimization.Trial.profile], which will automatically
    profile the block of code for memory before and after as well as time taken.

    If you've [`profile()`][amltk.optimization.Trial.profile]'ed any intervals,
    you can access them by name through
    [`trial.profiles`][amltk.optimization.Trial.profiles].
    Please see the [`Profiler`][amltk.profiling.profiler.Profiler]
    for more.

    ??? example "Profiling with a trial."

        ```python exec="true" source="material-block" result="python" title="profile"
        from amltk.optimization import Trial

        trial = Trial.create(name="some-unique-name", config={})

        # ... somewhere where you've begun your trial.
        with trial.profile("some_interval"):
            for work in range(100):
                pass

        print(trial.profiler.df())
        ```

    You can also record anything you'd like into the
    [`.summary`][amltk.optimization.Trial.summary], a plain `#!python dict`
    or use [`trial.store()`][amltk.optimization.Trial.store] to store artifacts
    related to the trial.

    ??? tip "What to put in `.summary`?"

        For large items, e.g. predictions or models, these are highly advised to
        [`.store()`][amltk.optimization.Trial.store] to disk, especially if using
        a `Task` for multiprocessing.

        Further, if serializing the report using the
        [`report.df()`][amltk.optimization.Trial.Report.df],
        returning a single row,
        or a [`History`][amltk.optimization.History]
        with [`history.df()`][amltk.optimization.History.df] for a dataframe consisting
        of many of the reports, then you'd likely only want to store things
        that are scalar and can be serialised to disk by a pandas DataFrame.

    """

    name: str
    """The unique name of the trial."""

    config: Mapping[str, Any]
    """The config of the trial provided by the optimizer."""

    bucket: PathBucket
    """The bucket to store trial related output to."""

    info: I | None = field(repr=False)
    """The info of the trial provided by the optimizer."""

    metrics: MetricCollection
    """The metrics associated with the trial.

    You can access the metrics by name, e.g. `#!python trial.metrics["loss"]`.
    """

    seed: int | None = None
    """The seed to use if suggested by the optimizer."""

    fidelities: Mapping[str, Any]
    """The fidelities at which to evaluate the trial, if any."""

    profiler: Profiler = field(repr=False)
    """A profiler for this trial."""

    summary: MutableMapping[str, Any]
    """The summary of the trial. These are for summary statistics of a trial and
    are single values."""

    storage: set[Any]
    """Anything stored in the trial, the elements of the list are keys that can be
    used to retrieve them later, such as a Path.
    """

    extras: MutableMapping[str, Any]
    """Any extras attached to the trial."""

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        name: str,
        config: Mapping[str, Any] | None = None,
        *,
        metrics: Metric | Iterable[Metric] | Mapping[str, Metric] | None = None,
        info: I | None = None,
        seed: int | None = None,
        fidelities: Mapping[str, Any] | None = None,
        profiler: Profiler | None = None,
        bucket: str | Path | PathBucket | None = None,
        summary: MutableMapping[str, Any] | None = None,
        storage: set[Hashable] | None = None,
        extras: MutableMapping[str, Any] | None = None,
    ) -> Trial[I]:
        """Create a trial.

        Args:
            name: The name of the trial.
            metrics: The metrics of the trial.
            config: The config of the trial.
            info: The info of the trial.
            seed: The seed of the trial.
            fidelities: The fidelities of the trial.
            bucket: The bucket of the trial.
            profiler: The profiler of the trial.
            summary: The summary of the trial.
            storage: The storage of the trial.
            extras: The extras of the trial.

        Returns:
            The trial.
        """
        return Trial(
            name=name,
            metrics=(
                MetricCollection.from_collection(metrics)
                if metrics is not None
                else MetricCollection()
            ),
            profiler=(
                profiler
                if profiler is not None
                else Profiler(memory_unit="B", time_kind="wall")
            ),
            config=config if config is not None else {},
            info=info,
            seed=seed,
            fidelities=fidelities if fidelities is not None else {},
            bucket=(
                bucket
                if isinstance(bucket, PathBucket)
                else (
                    PathBucket(bucket)
                    if bucket is not None
                    else PathBucket(f"trial-{name}-{datetime.now().isoformat()}")
                )
            ),
            summary=summary if summary is not None else {},
            storage=storage if storage is not None else set(),
            extras=extras if extras is not None else {},
        )

    @property
    def profiles(self) -> Mapping[str, Profile.Interval]:
        """The profiles of the trial."""
        return self.profiler.profiles

    def dump_exception(
        self,
        exception: BaseException,
        *,
        name: str | None = None,
    ) -> None:
        """Dump an exception to the trial.

        Args:
            exception: The exception to dump.
            name: The name of the file to dump to. If `None`, will be `"exception"`.
        """
        fname = name if name is not None else "exception"
        traceback = "".join(traceback_module.format_tb(exception.__traceback__))
        msg = f"{traceback}\n{exception.__class__.__name__}: {exception}"
        self.store({f"{fname}.txt": msg})

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        time: Timer.Kind | Literal["wall", "cpu", "process"] | None = None,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] | None = None,
        summary: bool = False,
    ) -> Iterator[None]:
        """Measure some interval in the trial.

        The results of the profiling will be available in the `.summary` attribute
        with the name of the interval as the key.

        ```python exec="true" source="material-block" result="python" title="profile"
        from amltk.optimization import Trial
        import time

        trial = Trial.create(name="trial", config={"x": 1})

        with trial.profile("some_interval"):
            # Do some work
            time.sleep(1)

        print(trial.profiler["some_interval"].time)
        ```

        Args:
            name: The name of the interval.
            time: The timer kind to use for the trial. Defaults to the default
                timer kind of the profiler.
            memory_unit: The memory unit to use for the trial. Defaults to the
                default memory unit of the profiler.
            summary: Whether to add the interval to the summary.

        Yields:
            The interval measured. Values will be nan until the with block is finished.
        """
        with self.profiler(name=name, memory_unit=memory_unit, time_kind=time):
            yield

        if summary:
            profile = self.profiler[name]
            self.summary.update(profile.to_dict(prefix=name))

    def success(self, **metrics: float | int) -> Trial.Report[I]:
        """Generate a success report.

        ```python exec="true" source="material-block" result="python" title="success" hl_lines="7"
        from amltk.optimization import Trial, Metric

        loss_metric = Metric("loss", minimize=True)

        trial = Trial.create(name="trial", config={"x": 1}, metrics=[loss_metric])
        report = trial.success(loss=1)

        print(report)
        ```

        Args:
            **metrics: The metrics of the trial, where the key is the name of the
                metrics and the value is the metric.

        Returns:
            The report of the trial.
        """  # noqa: E501
        values: dict[str, float] = {}

        for metric_def in self.metrics.values():
            if (reported_value := metrics.get(metric_def.name)) is not None:
                values[metric_def.name] = reported_value
            else:
                raise ValueError(
                    f" Please provide a value for the metric '{metric_def.name}' as "
                    " this is one of the metrics of the trial. "
                    f"\n Try `trial.success({metric_def.name}=value, ...)`.",
                )

        # Need to check if anything extra was reported!
        extra = set(metrics.keys()) - self.metrics.keys()
        if extra:
            raise ValueError(
                f"Cannot report `success()` with extra metrics: {extra=}."
                f"\nOnly metrics {list(self.metrics)} as these are the metrics"
                " provided for this trial."
                "\nTo record other numerics, use `trial.summary` instead.",
            )

        return Trial.Report(trial=self, status=Trial.Status.SUCCESS, values=values)

    def fail(
        self,
        exception: Exception | None = None,
        traceback: str | None = None,
        /,
        **metrics: float | int,
    ) -> Trial.Report[I]:
        """Generate a failure report.

        !!! note "Non specifed metrics"

            If you do not specify metrics, this will use
            the [`.metrics`][amltk.optimization.Trial.metrics] to determine
            the [`.worst`][amltk.optimization.Metric.worst] value of the metric,
            using that as the reported result

        ```python exec="true" source="material-block" result="python" title="fail"
        from amltk.optimization import Trial, Metric

        loss = Metric("loss", minimize=True, bounds=(0, 1_000))
        trial = Trial.create(name="trial", config={"x": 1}, metrics=[loss])

        try:
            raise ValueError("This is an error")  # Something went wrong
        except Exception as error:
            report = trial.fail(error)

        print(report.values)
        print(report)
        ```

        Returns:
            The result of the trial.
        """
        if exception is not None and traceback is None:
            traceback = traceback_module.format_exc()

        # Need to check if anything extra was reported!
        extra = set(metrics.keys()) - self.metrics.keys()
        if extra:
            raise ValueError(
                f"Cannot report `fail()` with extra metrics: {extra=}."
                f"\nOnly metrics {list(self.metrics)} as these are the metrics"
                " provided for this trial."
                "\nTo record other numerics, use `trial.summary` instead.",
            )

        return Trial.Report(
            trial=self,
            status=Trial.Status.FAIL,
            exception=exception,
            traceback=traceback,
            values=metrics,
        )

    def crashed(
        self,
        exception: Exception,
        traceback: str | None = None,
    ) -> Trial.Report[I]:
        """Generate a crash report.

        !!! note

            You will typically not create these manually, but instead if we don't
            recieve a report from a target function evaluation, but only an error,
            we assume something crashed and generate a crash report for you.

        !!! note "Non specifed metrics"

            We will use the [`.metrics`][amltk.optimization.Trial.metrics] to determine
            the [`.worst`][amltk.optimization.Metric.worst] value of the metric,
            using that as the reported metrics

        Args:
            exception: The exception that caused the crash. If not provided, the
                exception will be taken from the trial. If this is still `None`,
                a `RuntimeError` will be raised.
            traceback: The traceback of the exception. If not provided, the
                traceback will be taken from the trial if there is one there.

        Returns:
            The report of the trial.
        """
        if traceback is None:
            traceback = "".join(traceback_module.format_tb(exception.__traceback__))

        return Trial.Report(
            trial=self,
            status=Trial.Status.CRASHED,
            exception=exception,
            traceback=traceback,
        )

    def store(self, items: Mapping[str, T]) -> None:
        """Store items related to the trial.

        ```python exec="true" source="material-block" result="python" title="store" hl_lines="5"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        trial = Trial.create(name="trial", config={"x": 1}, bucket=PathBucket("my-trial"))
        trial.store({"config.json": trial.config})
        print(trial.storage)
        ```

        Args:
            items: The items to store, a dict from the key to store it under
                to the item itself.If using a `str`, `Path` or `PathBucket`,
                the keys of the items should be a valid filename, including
                the correct extension. e.g. `#!python {"config.json": trial.config}`
        """  # noqa: E501
        self.bucket.store(items)
        # Add the keys to storage
        self.storage.update(items)

    def delete_from_storage(self, items: Iterable[str]) -> dict[str, bool]:
        """Delete items related to the trial.

        ```python exec="true" source="material-block" result="python" title="delete-storage" hl_lines="6"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")
        trial = Trial.create(name="trial", config={"x": 1}, info={}, bucket=bucket)

        trial.store({"config.json": trial.config})
        trial.delete_from_storage(items=["config.json"])

        print(trial.storage)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="delete-storage-bucket" hl_lines="9"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")
        trial = Trial.create(name="trial", config={"x": 1}, bucket=bucket)

        trial.store({"config.json": trial.config})
        trial.delete_from_storage(items=["config.json"])

        print(trial.storage)
        ```

        Args:
            items: The items to delete, an iterable of keys

        Returns:
            A dict from the key to whether it was deleted or not.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        removed = self.bucket.remove(items)
        self.storage.difference_update(items)
        return removed

    def copy(self) -> Self:
        """Create a copy of the trial.

        Returns:
            The copy of the trial.
        """
        return copy.deepcopy(self)

    @overload
    def retrieve(self, key: str, *, check: None = None) -> Any:
        ...

    @overload
    def retrieve(self, key: str, *, check: type[R]) -> R:
        ...

    def retrieve(self, key: str, *, check: type[R] | None = None) -> R | Any:
        """Retrieve items related to the trial.

        ```python exec="true" source="material-block" result="python" title="retrieve" hl_lines="7"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        # Create a trial, normally done by an optimizer
        trial = Trial.create(name="trial", config={"x": 1}, bucket=bucket)

        trial.store({"config.json": trial.config})
        config = trial.retrieve("config.json")

        print(config)
        ```

        Args:
            key: The key of the item to retrieve as said in `.storage`.
            check: If provided, will check that the retrieved item is of the
                provided type. If not, will raise a `TypeError`.

        Returns:
            The retrieved item.

        Raises:
            TypeError: If `check=` is provided and  the retrieved item is not of the provided
                type.
        """  # noqa: E501
        return self.bucket[key].load(check=check)

    def attach_extra(self, name: str, plugin_item: Any) -> None:
        """Attach a plugin item to the trial.

        Args:
            name: The name of the plugin item.
            plugin_item: The plugin item.
        """
        self.extras[name] = plugin_item

    def rich_renderables(self) -> Iterable[RenderableType]:
        """The renderables for rich for this report."""
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.table import Table

        items: list[RenderableType] = []
        table = Table.grid(padding=(0, 1), expand=False)

        # Predfined things
        table.add_row("config", Pretty(self.config))

        if self.fidelities:
            table.add_row("fidelities", Pretty(self.fidelities))

        if any(self.extras):
            table.add_row("extras", Pretty(self.extras))

        if self.seed:
            table.add_row("seed", Pretty(self.seed))

        if self.bucket:
            table.add_row("bucket", Pretty(self.bucket))

        if self.metrics:
            items.append(
                Panel(Pretty(self.metrics), title="Metrics", title_align="left"),
            )

        # Dynamic things
        if self.summary:
            table.add_row("summary", Pretty(self.summary))

        if any(self.storage):
            table.add_row("storage", Pretty(self.storage))

        for name, profile in self.profiles.items():
            table.add_row("profile:" + name, Pretty(profile))

        items.append(table)

        yield from items

    @override
    def __rich__(self) -> RenderableType:
        from rich.console import Group as RichGroup
        from rich.panel import Panel
        from rich.text import Text

        title = Text.assemble(
            ("Trial", "bold"),
            ("(", "default"),
            (self.name, "italic"),
            (")", "default"),
        )

        return Panel(
            RichGroup(*self.rich_renderables()),
            title=title,
            title_align="left",
        )

    class Status(str, Enum):
        """The status of a trial."""

        SUCCESS = "success"
        """The trial was successful."""

        FAIL = "fail"
        """The trial failed."""

        CRASHED = "crashed"
        """The trial crashed."""

        UNKNOWN = "unknown"
        """The status of the trial is unknown."""

        @override
        def __str__(self) -> str:
            return self.value

        def __rich__(self) -> Text:
            from rich.text import Text

            styles = {
                Trial.Status.SUCCESS: "bold green",
                Trial.Status.FAIL: "bold yellow",
                Trial.Status.CRASHED: "bold red",
                Trial.Status.UNKNOWN: "bold underline",
            }

            return Text(self.value, style=styles.get(self, "bold underline"))

    @dataclass
    class Report(RichRenderable, Generic[I2]):
        """The [`Trial.Report`][amltk.optimization.Trial.Report] encapsulates
        a [`Trial`][amltk.optimization.Trial], its status and any metrics/exceptions
        that may have occured.

        Typically you will not create these yourself, but instead use
        [`trial.success()`][amltk.optimization.Trial.success] or
        [`trial.fail()`][amltk.optimization.Trial.fail] to generate them.

        ```python exec="true" source="material-block" result="python"
        from amltk.optimization import Trial, Metric

        loss = Metric("loss", minimize=True)

        trial = Trial.create(name="trial", config={"x": 1}, metrics=[loss])

        with trial.profile("fitting"):
            # Do some work
            # ...
            report = trial.success(loss=1)

        print(report.df())
        ```

        These reports are used to report back metrics to an
        [`Optimizer`][amltk.optimization.Optimizer]
        with [`Optimizer.tell()`][amltk.optimization.Optimizer.tell] but can also be
        stored for your own uses.

        You can access the original trial with the
        [`.trial`][amltk.optimization.Trial.Report.trial] attribute, and the
        [`Status`][amltk.optimization.Trial.Status] of the trial with the
        [`.status`][amltk.optimization.Trial.Report.status] attribute.

        You may also want to check out the [`History`][amltk.optimization.History] class
        for storing a collection of `Report`s, allowing for an easier time to convert
        them to a dataframe or perform some common Hyperparameter optimization parsing
        of metrics.
        """

        trial: Trial[I2]
        """The trial that was run."""

        status: Trial.Status
        """The status of the trial."""

        reported_at: datetime = field(default_factory=datetime.now)
        """When this Report was generated.

        This will primarily be `None` if there was no corresponding key
        when loading this report from a serialized form, such as
        with [`from_df()`][amltk.optimization.Trial.Report.from_df]
        or [`from_dict()`][amltk.optimization.Trial.Report.from_dict].
        """

        exception: BaseException | None = None
        """The exception reported if any."""

        traceback: str | None = field(repr=False, default=None)
        """The traceback reported if any."""

        values: Mapping[str, float] = field(default_factory=dict)
        """The reported metric values of the trial."""

        @property
        def name(self) -> str:
            """The name of the trial."""
            return self.trial.name

        @property
        def config(self) -> Mapping[str, Any]:
            """The config of the trial."""
            return self.trial.config

        @property
        def metrics(self) -> MetricCollection:
            """The metrics of the trial."""
            return self.trial.metrics

        @property
        def profiles(self) -> Mapping[str, Profile.Interval]:
            """The profiles of the trial."""
            return self.trial.profiles

        @property
        def summary(self) -> MutableMapping[str, Any]:
            """The summary of the trial."""
            return self.trial.summary

        @property
        def storage(self) -> set[str]:
            """The storage of the trial."""
            return self.trial.storage

        @property
        def bucket(self) -> PathBucket:
            """The bucket attached to the trial."""
            return self.trial.bucket

        @property
        def info(self) -> I2 | None:
            """The info of the trial, specific to the optimizer that issued it."""
            return self.trial.info

        def df(
            self,
            *,
            profiles: bool = True,
            configs: bool = True,
            summary: bool = True,
            metrics: bool = True,
        ) -> pd.DataFrame:
            """Get a dataframe of the trial.

            !!! note "Prefixes"

                * `summary`: Entries will be prefixed with `#!python "summary:"`
                * `config`: Entries will be prefixed with `#!python "config:"`
                * `storage`: Entries will be prefixed with `#!python "storage:"`
                * `metrics`: Entries will be prefixed with `#!python "metrics:"`
                * `profile:<name>`: Entries will be prefixed with
                    `#!python "profile:<name>:"`

            Args:
                profiles: Whether to include the profiles.
                configs: Whether to include the configs.
                summary: Whether to include the summary.
                metrics: Whether to include the metrics.
            """
            items = {
                "name": self.name,
                "status": str(self.status),
                "trial_seed": self.trial.seed if self.trial.seed else np.nan,
                "exception": str(self.exception) if self.exception else "NA",
                "traceback": str(self.traceback) if self.traceback else "NA",
                "bucket": str(self.bucket.path),
                "reported_at": self.reported_at,
            }
            if metrics:
                for metric_name, value in self.values.items():
                    metric_def = self.metrics[metric_name]
                    items[f"metric:{metric_def}"] = value
            if summary:
                items.update(**prefix_keys(self.trial.summary, "summary:"))
            if configs:
                items.update(**prefix_keys(self.trial.config, "config:"))
            if profiles:
                for name, profile in sorted(self.profiles.items(), key=lambda x: x[0]):
                    items.update(profile.to_dict(prefix=f"profile:{name}"))

            return pd.DataFrame(items, index=[0]).convert_dtypes().set_index("name")

        @overload
        def retrieve(self, key: str, *, check: None = None) -> Any:
            ...

        @overload
        def retrieve(self, key: str, *, check: type[R]) -> R:
            ...

        def retrieve(self, key: str, *, check: type[R] | None = None) -> R | Any:
            """Retrieve items related to the trial.

            ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

            from amltk.optimization import Trial
            from amltk.store import PathBucket

            bucket = PathBucket("results")

            trial = Trial.create(name="trial", config={"x": 1}, bucket=bucket)

            trial.store({"config.json": trial.config})
            report = trial.success()

            config = report.retrieve("config.json")
            print(config)
            ```

            Args:
                key: The key of the item to retrieve as said in `.storage`.
                check: If provided, will check that the retrieved item is of the
                    provided type. If not, will raise a `TypeError`.

            Returns:
                The retrieved item.

            Raises:
                TypeError: If `check=` is provided and  the retrieved item is not of the provided
                    type.
            """  # noqa: E501
            return self.trial.retrieve(key, check=check)

        def store(self, items: Mapping[str, T]) -> None:
            """Store items related to the trial.

            See Also:
                * [`Trial.store()`][amltk.optimization.trial.Trial.store]
            """
            self.trial.store(items)

        @classmethod
        def from_df(cls, df: pd.DataFrame | pd.Series) -> Trial.Report:
            """Create a report from a dataframe.

            See Also:
                * [`.from_dict()`][amltk.optimization.Trial.Report.from_dict]
            """
            if isinstance(df, pd.DataFrame):
                if len(df) != 1:
                    raise ValueError(
                        f"Expected a dataframe with one row, got {len(df)} rows.",
                    )
                series = df.iloc[0]
            else:
                series = df

            data_dict = {"name": series.name, **series.to_dict()}
            return cls.from_dict(data_dict)

        @classmethod
        def from_dict(cls, d: Mapping[str, Any]) -> Trial.Report:
            """Create a report from a dictionary.

            !!! note "Prefixes"

                Please see [`.df()`][amltk.optimization.Trial.Report.df]
                for information on what the prefixes should be for certain fields.

            Args:
                d: The dictionary to create the report from.

            Returns:
                The created report.
            """
            prof_dict = mapping_select(d, "profile:")
            if any(prof_dict):
                profile_names = sorted(
                    {name.rsplit(":", maxsplit=2)[0] for name in prof_dict},
                )
                profiles = {
                    name: Profile.from_dict(mapping_select(prof_dict, f"{name}:"))
                    for name in profile_names
                }
            else:
                profiles = {}

            # NOTE: We assume the order of the objectives are in the right
            # order in the dict. If we attempt to force a sort-order, we may
            # deserialize incorrectly. By not having a sort order, we rely
            # on serialization to keep the order, which is not ideal either.
            # May revisit this if we need to
            raw_metrics: dict[str, float] = mapping_select(d, "metric:")
            metrics: dict[Metric, float] = {
                Metric.from_str(name): value for name, value in raw_metrics.items()
            }

            exception = d.get("exception")
            traceback = d.get("traceback")
            trial_seed = d.get("trial_seed")
            if pd.isna(exception) or exception == "NA":  # type: ignore
                exception = None
            if pd.isna(traceback) or traceback == "NA":  # type: ignore
                traceback = None
            if pd.isna(trial_seed):  # type: ignore
                trial_seed = None

            if (_bucket := d.get("bucket")) is not None:
                bucket = PathBucket(_bucket)
            else:
                bucket = PathBucket(f"uknown_trial_bucket-{datetime.now().isoformat()}")

            trial: Trial = Trial.create(
                name=d["name"],
                config=mapping_select(d, "config:"),
                info=None,  # We don't save this to disk so we load it back as None
                bucket=bucket,
                seed=trial_seed,
                fidelities=mapping_select(d, "fidelities:"),
                profiler=Profiler(profiles=profiles),
                metrics=metrics.keys(),
                summary=mapping_select(d, "summary:"),
                storage=set(mapping_select(d, "storage:").values()),
                extras=mapping_select(d, "extras:"),
            )
            _values: dict[str, float] = {m.name: v for m, v in metrics.items()}

            status = Trial.Status(dict_get_not_none(d, "status", "unknown"))
            match status:
                case Trial.Status.SUCCESS:
                    report = trial.success(**_values)
                case Trial.Status.FAIL:
                    exc = Exception(exception) if exception else None
                    tb = str(traceback) if traceback else None
                    report = trial.fail(exc, tb, **_values)
                case Trial.Status.CRASHED:
                    exc = Exception(exception) if exception else Exception("Unknown")
                    tb = str(traceback) if traceback else None
                    report = trial.crashed(exc, tb)
                case Trial.Status.UNKNOWN | _:
                    report = trial.crashed(exception=Exception("Unknown status."))

            timestamp = d.get("reported_at")
            if timestamp is None:
                raise ValueError(
                    "Cannot load report from dict without a 'reported_at' field.",
                )
            report.reported_at = parse_timestamp_object(timestamp)
            return report

        def rich_renderables(self) -> Iterable[RenderableType]:
            """The renderables for rich for this report."""
            from rich.pretty import Pretty
            from rich.text import Text

            yield Text.assemble(
                ("Status", "bold"),
                ("(", "default"),
                self.status.__rich__(),
                (")", "default"),
            )
            yield Pretty(self.metrics)
            yield from self.trial.rich_renderables()

        @override
        def __rich__(self) -> Panel:
            from rich.console import Group as RichGroup
            from rich.panel import Panel
            from rich.text import Text

            title = Text.assemble(
                ("Report", "bold"),
                ("(", "default"),
                (self.name, "italic"),
                (") - ", "default"),
                self.status.__rich__(),
            )
            return Panel(RichGroup(*self.rich_renderables()), title=title)
