"""A trial for an optimization task.

TODO: Populate more here.
"""
from __future__ import annotations

import logging
import traceback
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
from typing_extensions import Concatenate, ParamSpec

import numpy as np
import pandas as pd

from amltk.events import Event, Subscriber, funcname
from amltk.functional import mapping_select, prefix_keys
from amltk.scheduling.task import Task as TaskBase
from amltk.store import Bucket, PathBucket
from amltk.timing import TimeInterval, TimeKind, Timer

if TYPE_CHECKING:
    from concurrent.futures import Future

    from amltk.scheduling.scheduler import Scheduler
    from amltk.scheduling.task_plugin import TaskPlugin


Info = TypeVar("Info")
InfoInner = TypeVar("InfoInner")

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class Trial(Generic[Info]):
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

    Attributes:
        summary: The summary of the trial. These are for
            summary statistics of a trial and are single values.
        stored: Anything stored in the trial, the elements
            of the list are keys that can be used to retrieve them
            later, such as a Path.
        time: The time taken by the trial, once ended.
        timer: The timer used to time the trial, once begun.
        exception: The exception raised by the trial, if any.
        traceback: The traceback of the exception, if any.
        plugins: Any plugins attached to the trial.
    """

    name: str
    config: Mapping[str, Any]
    info: Info = field(repr=False)
    seed: int | None = None
    fidelities: dict[str, Any] | None = None

    time: TimeInterval | None = field(repr=False, default=None)
    timer: Timer | None = field(repr=False, default=None)

    summary: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    exception: Exception | None = field(repr=True, default=None)
    traceback: str | None = field(repr=False, default=None)

    storage: set[Any] = field(default_factory=set)
    plugins: dict[str, Any] = field(default_factory=dict)

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

    def success(self, **results: Any) -> Trial.SuccessReport[Info]:
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

    def fail(self, **results: Any) -> Trial.FailReport[Info]:
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
    ) -> Trial.CrashReport[Info]:
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

    @dataclass
    class Report(Generic[InfoInner]):
        """A report for a trial.

        !!! note "Specific Instansiations"

            * [`SuccessReport`][amltk.optimization.Trial.SuccessReport] -
                Created with [`success()`][amltk.optimization.Trial.success]

            * [`FailReport`][amltk.optimization.Trial.FailReport] -
                Created with [`fail()`][amltk.optimization.Trial.fail]

            * [`CrashReport`][amltk.optimization.Trial.CrashReport] -
                Created with [`crashed()`][amltk.optimization.Trial.crashed]


        Attributes:
            trial: The trial that was run.
        """

        exception: BaseException | None
        results: dict[str, Any] | None
        time: TimeInterval | None
        traceback: str | None
        trial: Trial[InfoInner]

        status: str

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
        def info(self) -> InfoInner | None:
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
    class CrashReport(Report[InfoInner]):
        """A report for a crashed trial."""

        exception: BaseException
        results: None
        time: TimeInterval
        traceback: str | None
        trial: Trial[InfoInner]

        status: str = "crashed"

    @dataclass
    class SuccessReport(Report[InfoInner]):
        """A report for a successful trial."""

        exception: None
        results: dict[str, Any]
        time: TimeInterval
        traceback: None
        trial: Trial[InfoInner]

        status: str = "success"

    @dataclass
    class FailReport(Report[InfoInner]):
        """A report for a failed trial."""

        exception: BaseException | None
        results: dict[str, Any]
        time: TimeInterval
        traceback: str | None
        trial: Trial[InfoInner]

        status: str = "fail"

    class Objective(Generic[P, InfoInner]):
        """Attach static information to a function to be optimized."""

        def __init__(
            self,
            f: Callable[Concatenate[Trial[InfoInner], P], Trial.Report[InfoInner]],
            *args: P.args,
            **kwargs: P.kwargs,
        ):
            """Initialize the objective.

            Args:
                f: The function to optimize.
                args: The positional arguments to pass to `f` after trial.
                kwargs: The keyword arguments to pass to `f`.
            """
            self.f = f
            self.args = args
            self.kwargs = kwargs

        def __call__(self, trial: Trial[InfoInner]) -> Trial.Report[InfoInner]:
            """Call the objective."""
            return self.f(trial, *self.args, **self.kwargs)

    class Task(TaskBase):
        """A Task specifically for Trials."""

        SUCCESS: Event[Trial.SuccessReport] = Event("trial-success")
        """The event that is triggered when a trial succeeds."""

        FAILURE: Event[Trial.FailReport] = Event("trial-failure")
        """The event that is triggered when a trial fails."""

        CRASHED: Event[Trial.CrashReport] = Event("trial-crashed")
        """The event that is triggered when a trial crashes."""

        REPORT: Event[Trial.Report] = Event("trial-report")
        """The event that is triggered when a trial reports anything."""

        def __init__(
            self,
            function: Callable[[Trial[InfoInner]], Trial.Report[InfoInner]],
            scheduler: Scheduler,
            *,
            name: str | None = None,
            plugins: Iterable[
                TaskPlugin[[Trial[InfoInner]], Trial.Report[InfoInner]]
            ] = (),
        ) -> None:
            """Initialize a task.

            See [`Task`][amltk.scheduling.task.Task] for more details.

            Args:
                function: The function to run.
                scheduler: The scheduler to use.
                name: The name of the task.
                plugins: Any plugins to attach to the task.
            """
            # NOTE: It's important these are here to setup up the subscribers
            # properly
            self.name = funcname(function) if name is None else name
            self.scheduler = scheduler

            self.on_report: Subscriber[Trial.Report[InfoInner]]
            self.on_report = self.subscriber(self.REPORT)

            self.on_failed: Subscriber[Trial.FailReport[InfoInner]]
            self.on_failed = self.subscriber(self.FAILURE)

            self.on_success: Subscriber[Trial.SuccessReport[InfoInner]]
            self.on_success = self.subscriber(self.SUCCESS)

            self.on_crashed: Subscriber[Trial.CrashReport[InfoInner]]
            self.on_crashed = self.subscriber(self.CRASHED)

            self.trial_lookup: dict[Future, Trial] = {}

            super().__init__(function, scheduler, name=name, plugins=plugins)

            self.on_f_returned(self._emit_report)
            self.on_f_exception(self._emit_report)
            self.on_f_submitted(self._register_future)

        def _emit_report(
            self,
            future: Future,
            report: Trial.Report | BaseException,
        ) -> None:
            """Emit a report for a trial based on the type of the report."""
            # Emit the fact a report happened
            if isinstance(report, BaseException):
                trial = self.trial_lookup.get(future)
                if trial is None:
                    logger.error(f"No trial found for future {future}!")
                    return

                report = trial.crashed(report)

            emit_items: dict[Event, Any] = {
                self.REPORT: ((report,), None),
            }

            # Emit the specific type of report
            event: Event
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

        def _register_future(
            self,
            future: Future[Any],
            trial: Trial,
            *args: Any,  # noqa: ARG002
            **kwargs: Any,  # noqa: ARG002
        ) -> None:
            self.trial_lookup[future] = trial
