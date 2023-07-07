"""A trial for an optimization task.

TODO: Populate more here.
"""
from __future__ import annotations

import copy
import logging
import traceback
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    overload,
)
from typing_extensions import Concatenate, ParamSpec, Self

import numpy as np
import pandas as pd

from amltk.events import Emitter, Event, Subscriber
from amltk.functional import mapping_select, prefix_keys
from amltk.scheduling.task import (
    Task as TaskBase,
)
from amltk.store import Bucket, PathBucket
from amltk.timing import TimeInterval, TimeKind, Timer

if TYPE_CHECKING:
    from concurrent.futures import Future

    from amltk.scheduling.task import (
        Scheduler,
        TaskPlugin,
    )


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


@dataclass
class Trial(Generic[I]):
    """A trial as suggested by an optimizer.

    You can modify the Trial as you see fit, specifically
    `.summary` which are for recording any information you may like.

    The other attributes will be automatically set, such
    as `.time`, `.timer` and `.exception`, which are capture
    using [`trial.begin()`][amltk.optimization.trial.Trial.begin].

    Args:
        name: The unique name of the trial.
        config: The config for the trial.
        info: The info of the trial.
        seed: The seed to use if suggested by the optimizer.
    """

    name: str
    """The unique name of the trial."""

    config: Mapping[str, Any]
    """The config of the trial provided by the optimizer."""

    info: I = field(repr=False)
    """The info of the trial provided by the optimizer."""

    seed: int | None = None
    """The seed to use if suggested by the optimizer."""

    fidelities: dict[str, Any] | None = None
    """The fidelities at which to evaluate the trial, if any."""

    time: TimeInterval | None = field(repr=False, default=None)
    """The time taken by the trial, once ended."""

    timer: Timer | None = field(repr=False, default=None)
    """The timer used to time the trial, once begun."""

    summary: dict[str, Any] = field(default_factory=dict)
    """The summary of the trial. These are for summary statistics of a trial and
    are single values."""

    results: dict[str, Any] = field(default_factory=dict)
    """The results of the trial. These are what will be reported to the optimizer.
    These are mainly set by the [`success()`][amltk.optimization.trial.Trial.success]
    and [`fail()`][amltk.optimization.trial.Trial.fail] methods."""

    exception: Exception | None = field(repr=True, default=None)
    """The exception raised by the trial, if any."""

    traceback: str | None = field(repr=False, default=None)
    """The traceback of the exception, if any."""

    storage: set[Any] = field(default_factory=set)
    """Anything stored in the trial, the elements of the list are keys that can be
    used to retrieve them later, such as a Path.
    """

    plugins: dict[str, Any] = field(default_factory=dict)
    """Any plugins attached to the trial."""

    @contextmanager
    def begin(
        self,
        time: TimeKind | Literal["wall", "cpu", "process"] = "cpu",
    ) -> Iterator[None]:
        """Begin the trial with a `contextmanager`.

        Will begin timing the trial in the `with` block, attaching the timings to the
        trial once completed, under `.time` and the timer itself under `.timer`.

        If an exception is raised, it will be attached to the trial under `.exception`
        with the traceback attached to the actual error message, such that it can
        be pickled and sent back to the main process loop.

        ```python exec="true" source="material-block" result="python" title="begin" hl_lines="8 9 10"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        assert trial.time is None
        assert trial.timer is None

        with trial.begin():
            # Do some work
            pass

        print(trial.time)
        print(trial.timer)
        ```

        Args:
            time: The timer kind to use for the trial.
        """  # noqa: E501
        self.timer = Timer.start(kind=time)
        try:
            yield
        except Exception as error:  # noqa: BLE001
            self.exception = error
            self.traceback = traceback.format_exc()
        finally:
            if self.time is None:
                self.time = self.timer.stop()

    def success(self, **results: Any) -> Trial.SuccessReport[I]:
        """Generate a success report.

        ```python exec="true" source="material-block" result="python" title="success" hl_lines="7"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.begin():
            # Do some work
            report = trial.success(cost=1)

        print(report)
        ```

        Returns:
            The result of the trial.
        """  # noqa: E501
        if self.timer is None:
            raise RuntimeError(
                "Cannot succeed a trial that has not been started."
                " Please use `with trial.begin():` to start the trial.",
            )

        time = self.time if self.time is not None else self.timer.stop()
        self.results = results
        return Trial.SuccessReport(
            trial=self,
            time=time,
            results=results,
            exception=None,
            traceback=None,
        )

    def fail(self, **results: Any) -> Trial.FailReport[I]:
        """Generate a failure report.

        ```python exec="true" source="material-block" result="python" title="fail" hl_lines="6 9 10"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.begin():
            raise ValueError("This is an error")  # Something went wrong
            report = trial.success(cost=1)

        if trial.exception: # You can check for an exception of the trial here
            report = trial.fail(cost=100)

        print(report)
        ```

        Returns:
            The result of the trial.
        """  # noqa: E501
        if self.timer is None:
            raise RuntimeError(
                "Cannot fail a trial that has not been started."
                " Please use `with trial.begin():` to start the trial.",
            )

        time = self.time if self.time is not None else self.timer.stop()
        exception = self.exception
        traceback = self.traceback
        self.results = results

        return Trial.FailReport(
            trial=self,
            time=time,
            exception=exception,
            traceback=traceback,
            results=results,
        )

    def crashed(
        self,
        exception: BaseException | None = None,
        traceback: str | None = None,
    ) -> Trial.CrashReport[I]:
        """Generate a crash report.

        !!! note

            You will typically not create these manually, but instead if we don't
            recieve a report from a target function evaluation, but only an error,
            we assume something crashed and generate a crash report for you.

        Args:
            exception: The exception that caused the crash. If not provided, the
                exception will be taken from the trial. If this is still `None`,
                a `RuntimeError` will be raised.
            traceback: The traceback of the exception. If not provided, the
                traceback will be taken from the trial if there is one there.

        Returns:
            The result of the trial.
        """
        if exception is None and self.exception is None:
            raise RuntimeError(
                "Cannot generate a crash report without an exception."
                " Please provide an exception or use `with trial.begin():` to start"
                " the trial.",
            )

        exception = exception if exception else self.exception
        traceback = traceback if traceback else self.traceback
        assert exception is not None

        return Trial.CrashReport(
            trial=self,
            exception=exception,
            traceback=traceback,
            time=TimeInterval.na_time_interval(),
            results=None,
        )

    def store(
        self,
        items: Mapping[str, T],
        *,
        where: str | Path | Bucket | Callable[[str, Mapping[str, T]], None],
    ) -> None:
        """Store items related to the trial.

        ```python exec="true" source="material-block" result="python" title="store" hl_lines="5"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        print(trial.storage)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="store-bucket" hl_lines="8"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where=bucket)

        print(trial.storage)
        ```

        Args:
            items: The items to store, a dict from the key to store it under
                to the item itself. If using a `str`, `Path` or `PathBucket`,
                the keys of the items should be a valid filename, including
                the correct extension. e.g. `#!python {"config.json": trial.config}`

            where: Where to store the items.

                * If a `str` or `Path`, will store
                a bucket will be created at the path, and the items will be
                stored in a sub-bucket with the name of the trial.

                * If a `Bucket`, will store the items in a sub-bucket with the
                name of the trial.

                * If a `Callable`, will call the callable with the name of the
                trial and the key-valued pair of items to store.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        method: Callable[[str, dict[str, Any]], None] | Bucket
        if isinstance(where, str):
            where = Path(where)
            method = PathBucket(where, create=True)
        elif isinstance(where, Path):
            method = PathBucket(where, create=True)
        else:
            method = where

        if isinstance(method, Bucket):
            # Store in a sub-bucket
            method.sub(self.name).store(items)
        else:
            # Leave it up to supplied method
            method(self.name, items)

        # Add the keys to storage
        self.storage.update(items.keys())

    def copy(self) -> Self:
        """Create a copy of the trial.

        Returns:
            The copy of the trial.
        """
        return copy.deepcopy(self)

    @overload
    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: None = None,
    ) -> Any:
        ...

    @overload
    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: type[R],
    ) -> R:
        ...

    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: type[R] | None = None,
    ) -> R | Any:
        """Retrieve items related to the trial.

        !!! note "Same argument for `where=`"

             Use the same argument for `where=` as you did for `store()`.

        ```python exec="true" source="material-block" result="python" title="retrieve" hl_lines="7"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        config = trial.retrieve("config.json", where="./results")
        print(config)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where=bucket)

        config = trial.retrieve("config.json", where=bucket)
        print(config)
        ```

        Args:
            key: The key of the item to retrieve as said in `.storage`.
            check: If provided, will check that the retrieved item is of the
                provided type. If not, will raise a `TypeError`. This
                is only used if `where=` is a `str`, `Path` or `Bucket`.
            where: Where to retrieve the items from.

                * If a `str` or `Path`, will store
                a bucket will be created at the path, and the items will be
                retrieved from a sub-bucket with the name of the trial.

                * If a `Bucket`, will retrieve the items from a sub-bucket with the
                name of the trial.

        Returns:
            The retrieved item.

        Raises:
            TypeError: If `check=` is provided and  the retrieved item is not of the provided
                type.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        method: Bucket[str, Any]
        if isinstance(where, str):
            where = Path(where)
            method = PathBucket(where, create=True)
        elif isinstance(where, Path):
            method = PathBucket(where, create=True)
        else:
            method = where

        # Store in a sub-bucket
        return method.sub(self.name)[key].load(check=check)

    def attach_plugin_item(self, name: str, plugin_item: Any) -> None:
        """Attach a plugin item to the trial.

        Args:
            name: The name of the plugin item.
            plugin_item: The plugin item.
        """
        self.plugins[name] = plugin_item

    @classmethod
    def task(
        cls,
        function: Callable[Concatenate[Trial[T], P], Trial.Report[T]],
        scheduler: Scheduler,
        *,
        name: str | None = None,
        plugins: Iterable[TaskPlugin[Concatenate[Trial[T], P], Trial.Report[T]]] = (),
        init_plugins: bool = True,
    ) -> Trial.Task[T, P]:
        """Initialize a task.

        Uses the same arguments as [`Task`][amltk.Task].

        Args:
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            name: The name of the task.
            plugins: The plugins to use for this task.
            init_plugins: Whether to initialize the plugins or not.
        """
        task: TaskBase[Concatenate[Trial[T], P], Trial.Report[T]]
        task = TaskBase(
            function=function,
            scheduler=scheduler,
            name=name,
            plugins=plugins,  # type: ignore
            init_plugins=init_plugins,
        )
        return Trial.Task[T, P](task)

    @dataclass
    class Report(Generic[I2]):
        """A report for a trial.

        !!! note "Specific Instansiations"

            * [`SuccessReport`][amltk.optimization.Trial.SuccessReport] -
                Created with [`success()`][amltk.optimization.Trial.success]

            * [`FailReport`][amltk.optimization.Trial.FailReport] -
                Created with [`fail()`][amltk.optimization.Trial.fail]

            * [`CrashReport`][amltk.optimization.Trial.CrashReport] -
                Created with [`crashed()`][amltk.optimization.Trial.crashed]
        """

        results: dict[str, Any] | None
        """The results of the trial, if any."""

        time: TimeInterval | None
        """The time interval of the trial, if any."""

        exception: BaseException | None
        """The exception that was raised, if any."""

        traceback: str | None
        """The traceback of the exception, if any."""

        trial: Trial[I2]
        """The trial that was run."""

        status: str
        """The status of the trial."""

        DF_COLUMN_TYPES: ClassVar[dict] = {
            "name": pd.StringDtype(),
            "status": pd.StringDtype(),
            # As trial_seed can be None, we can't represent this in an int column.
            # Tried np.nan but that only works with floats
            "trial_seed": pd.Int64Dtype(),
            "exception": pd.StringDtype(),
            "traceback": pd.StringDtype(),
            "time:start": float,
            "time:end": float,
            "time:duration": float,
            "time:kind": pd.StringDtype(),
            "time:unit": pd.StringDtype(),
        }

        @property
        def name(self) -> str:
            """The name of the trial."""
            return self.trial.name

        @property
        def config(self) -> Mapping[str, Any]:
            """The config of the trial."""
            return self.trial.config

        @property
        def summary(self) -> dict[str, Any]:
            """The summary of the trial."""
            return self.trial.summary

        @property
        def storage(self) -> set[str]:
            """The storage of the trial."""
            return self.trial.storage

        @property
        def info(self) -> I2 | None:
            """The info of the trial, specific to the optimizer that issued it."""
            return self.trial.info

        def df(self) -> pd.DataFrame:
            """Get a dataframe of the results of the trial.

            !!! note "Prefixes"

                * `summary`: Entries will be prefixed with `#!python "summary:"`
                * `config`: Entries will be prefixed with `#!python "config:"`
                * `results`: Entries will be prefixed with `#!python "results:"`
                * `time`: Entries will be prefixed with `#!python "time:"`
                * `storage`: Entries will be prefixed with `#!python "storage:"`
            """
            _df = pd.DataFrame(
                {
                    "name": self.name,
                    "status": self.status,
                    "trial_seed": self.trial.seed if self.trial.seed else np.nan,
                    **prefix_keys(self.trial.summary, "summary:"),
                    **prefix_keys(self.results or {}, "results:"),
                    **prefix_keys(self.trial.config, "config:"),
                    **prefix_keys(
                        self.time.dict_for_dataframe() if self.time else {},
                        "time:",
                    ),
                    "exception": str(self.exception) if self.exception else None,
                    "traceback": self.traceback,
                },
                index=[0],
            )
            present_cols = {k: v for k, v in self.DF_COLUMN_TYPES.items() if k in _df}
            return _df.astype(present_cols)

        @overload
        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: None = None,
        ) -> Any:
            ...

        @overload
        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: type[R],
        ) -> R:
            ...

        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: type[R] | None = None,
        ) -> R | Any:
            """Retrieve items related to the trial.

            !!! note "Same argument for `where=`"

                 Use the same argument for `where=` as you did for `store()`.

            ```python exec="true" source="material-block" result="python" title="retrieve" hl_lines="7"
            from amltk.optimization import Trial

            trial = Trial(name="trial", config={"x": 1}, info={})

            trial.store({"config.json": trial.config}, where="./results")
            with trial.begin():
                report = trial.success()

            config = report.retrieve("config.json", where="./results")
            print(config)
            ```

            You could also create a Bucket and use that instead.

            ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

            from amltk.optimization import Trial
            from amltk.store import PathBucket

            bucket = PathBucket("results")

            trial = Trial(name="trial", config={"x": 1}, info={})

            trial.store({"config.json": trial.config}, where="./results")

            with trial.begin():
                report = trial.success()

            config = report.retrieve("config.json", where=bucket)
            print(config)
            ```

            Args:
                key: The key of the item to retrieve as said in `.storage`.
                check: If provided, will check that the retrieved item is of the
                    provided type. If not, will raise a `TypeError`. This
                    is only used if `where=` is a `str`, `Path` or `Bucket`.
                where: Where to retrieve the items from.

                    * If a `str` or `Path`, will store
                    a bucket will be created at the path, and the items will be
                    retrieved from a sub-bucket with the name of the trial.

                    * If a `Bucket`, will retrieve the items from a sub-bucket with the
                    name of the trial.

            Returns:
                The retrieved item.

            Raises:
                TypeError: If `check=` is provided and  the retrieved item is not of the provided
                    type.
            """  # noqa: E501
            # If not a Callable, we convert to a path bucket
            method: Bucket[str, Any]
            if isinstance(where, str):
                where = Path(where)
                method = PathBucket(where, create=True)
            elif isinstance(where, Path):
                method = PathBucket(where, create=True)
            else:
                method = where

            # Store in a sub-bucket
            return method.sub(self.name)[key].load(check=check)

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
                data_dict = df.iloc[0].to_dict()
            else:
                data_dict = df.to_dict()

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
            # The relative_ here is inserted in `df()` method of history, however
            # TimeInterval doesn't know about it, so we remove it here.
            time_interval = {
                k: v
                for k, v in mapping_select(d, "time:").items()
                if ("relative_" not in k and k != "duration")
            }
            fidelities = mapping_select(d, "fidelities:")
            config = mapping_select(d, "config:")
            summary = mapping_select(d, "summary:")
            results = mapping_select(d, "results:")

            trial: Trial[Any] = Trial(
                name=d["name"],
                config=config,
                info=None,  # We don't save this to disk so we load it back as None
                seed=d["trial_seed"],
                fidelities=fidelities,
                time=TimeInterval.from_dict(time_interval) if time_interval else None,
                timer=None,
                results=results,
                summary=summary,
                exception=d["exception"],
                traceback=d["traceback"],
            )
            return cls(
                status=d["status"],
                trial=trial,
                exception=d["exception"],
                traceback=d["traceback"],
                time=trial.time,
                results=results,
            )

    @dataclass
    class CrashReport(Report[I2]):
        """A report for a crashed trial."""

        exception: BaseException
        results: None
        time: TimeInterval
        traceback: str | None
        trial: Trial[I2]

        status: str = "crashed"

    @dataclass
    class SuccessReport(Report[I2]):
        """A report for a successful trial."""

        exception: None
        results: dict[str, Any]
        time: TimeInterval
        traceback: None
        trial: Trial[I2]

        status: str = "success"

    @dataclass
    class FailReport(Report[I2]):
        """A report for a failed trial."""

        exception: BaseException | None
        results: dict[str, Any]
        time: TimeInterval
        traceback: str | None
        trial: Trial[I2]

        status: str = "fail"

    class Task(Generic[I2, P2], Emitter):
        """A Task specifically for Trials."""

        on_report: Subscriber[Trial.Report[I2]]
        """
        A [`Subscriber`][amltk.Subscriber] called on a trial succeeds, fails or crashes.
        ```python
        @trial.on_report
        def on_report(report: Trial.Report):
            print(report)
        ```
        """

        on_failed: Subscriber[Trial.FailReport[I2]]
        """
        A [`Subscriber`][amltk.Subscriber] called when a trial reported as failed.
        ```python
        @trial.on_failed
        def on_report(report: Trial.FailReport):
            print(report)
        ```
        """

        on_success: Subscriber[Trial.SuccessReport[I2]]
        """
        A [`Subscriber`][amltk.Subscriber] called when a trial succeeds.
        ```python
        @trial.on_success
        def on_success(report: Trial.SuccessReport):
            print(report)
        ```
        """

        on_crashed: Subscriber[Trial.CrashReport[I2]]
        """
        A [`Subscriber`][amltk.Subscriber] called when a trial crashes and failed to
        report.
        ```python
        @trial.on_crashed
        def on_crashed(report: Trial.CrashReport):
            print(report)
        ```
        """
        on_cancelled: Subscriber[Trial.CrashReport[I2]]
        """
        A [`Subscriber`][amltk.Subscriber] called when a trial was cancelled. This
        will still return a [`CrashReport`][amltk.optimization.Trial.CrashReport],
        but should likely not be reported as a crash.
        ```python
        @trial.on_cancelled
        def on_cancelled(report: Trial.CrashReport):
            print(report)
        ```
        """

        SUCCESS: Event[Trial.SuccessReport[I2]] = Event("trial-success")
        FAILURE: Event[Trial.FailReport[I2]] = Event("trial-failure")
        CRASHED: Event[Trial.CrashReport[I2]] = Event("trial-crashed")
        REPORT: Event[Trial.Report[I2]] = Event("trial-report")
        CANCELLED: Event[Trial[I2]] = Event("trial-cancelled")

        def __init__(
            self,
            task: TaskBase[Concatenate[Trial[I2], P2], Trial.Report[I2]],
        ) -> None:
            """Initialize a task.

            See [`Task`][amltk.scheduling.task.Task] for more details.

            Args:
                task: The task to wrap.
            """
            super().__init__(event_manager=task.event_manager)
            self.on_report = self.subscriber(self.REPORT)
            self.on_failed = self.subscriber(self.FAILURE)
            self.on_success = self.subscriber(self.SUCCESS)
            self.on_crashed = self.subscriber(self.CRASHED)

            self.task = task
            self._trial_lookup: dict[Future, Trial] = {}

            self.task.on_f_returned(self._emit_report)
            self.task.on_f_exception(self._emit_report)
            self.task.on_f_cancelled(self._emit_report)

        def __call__(
            self,
            trial: Trial[I2],
            *args: P2.args,
            **kwargs: P2.kwargs,
        ) -> Future | None:
            """Submit a trial to the task.

            Args:
                trial: The trial to submit.
                args: The positional arguments to pass to the task.
                kwargs: The keyword arguments to pass to the task.

            Returns:
                The future for the trial.
            """
            future = self.task(trial, *args, **kwargs)
            if future is not None:
                self._trial_lookup[future] = trial

            return future

        def submit(
            self,
            trial: Trial[I2],
            *args: P2.args,
            **kwargs: P2.kwargs,
        ) -> Future | None:
            """Submit a trial to the task.

            Args:
                trial: The trial to submit.
                args: The positional arguments to pass to the task.
                kwargs: The keyword arguments to pass to the task.

            Returns:
                The future for the trial.
            """
            return self.__call__(trial, *args, **kwargs)

        def _emit_report(
            self,
            future: Future,
            report: Trial.Report | BaseException | None = None,
        ) -> None:
            """Emit a report for a trial based on the type of the report."""
            # If we didn't get a report, it means it was cancelled
            if report is None:
                report = CancelledError()
                was_cancelled = True
            else:
                was_cancelled = False

            if isinstance(report, BaseException):
                trial = self._trial_lookup.get(future)
                if trial is None:
                    logger.error(f"No trial found for future {future}!")
                    return

                report = trial.crashed(report)

            emit_items: dict[Event, Any] = {
                self.REPORT: ((report,), None),
            }

            if was_cancelled:
                emit_items[self.CANCELLED] = ((report,), None)

            # Emit the specific type of report
            event: Event
            if was_cancelled:
                event = self.CANCELLED
            if isinstance(report, Trial.SuccessReport):
                event = self.SUCCESS
            elif isinstance(report, Trial.FailReport):
                event = self.FAILURE
            elif isinstance(report, Trial.CrashReport):
                event = self.CRASHED
            else:
                raise TypeError(f"Unexpected report type: {type(report)}")

            emit_items[event] = ((report,), None)
            self.emit_many(emit_items)  # type: ignore
