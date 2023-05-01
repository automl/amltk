"""A trial for an optimization task.

TODO: Populate more here.
"""
from __future__ import annotations

from abc import ABC
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    overload,
)
from typing_extensions import Concatenate, ParamSpec

import pandas as pd

from byop.events import Event, Subscriber
from byop.exceptions import attach_traceback
from byop.functional import prefix_keys
from byop.scheduling import (
    Scheduler,
    Task as TaskBase,
)
from byop.store import Bucket, PathBucket
from byop.timing import TimeInterval, TimeKind, Timer
from byop.types import abstractmethod

if TYPE_CHECKING:
    from concurrent.futures import Future


Info = TypeVar("Info")
InfoInner = TypeVar("InfoInner")

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Trial(Generic[Info]):
    """A trial as suggested by an optimizer.

    You can modify the Trial as you see fit, specifically
    `.summary` and `.stats` which are for recording any
    information you may like.

    The other attributes will be automatically set, such
    as `.time`, `.timer` and `.exception`, which are capture
    using [`trial.begin()`][byop.optimization.trial.Trial.begin].

    Args:
        name: The unique name of the trial.
        config: The config for the trial.
        info: The info of the trial.
        seed: The seed to use if suggested by the optimizer.

    Attributes:
        summary: The summary of the trial. These are for
            summary statistics of a trial and are single values.
        stats: The stats of the trial. These are lists
            of values, where each value is the value of the stat
            in some order.
        stored: Anything stored in the trial, the elements
            of the list are keys that can be used to retrieve them
            later, such as a Path.
        time: The time taken by the trial, once ended.
        timer: The timer used to time the trial, once begun.
        exception: The exception raised by the trial, if any.
    """

    name: str
    config: Mapping[str, Any]
    info: Info = field(repr=False)
    seed: int | None = None

    time: TimeInterval | None = field(repr=False, default=None)
    timer: Timer | None = field(repr=False, default=None)
    exception: Exception | None = field(repr=False, default=None)

    summary: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list))
    storage: set[Any] = field(default_factory=set)
    results: dict[str, Any] = field(default_factory=dict)

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
        from byop.optimization import Trial

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
            self.exception = attach_traceback(error)
        finally:
            if self.time is None:
                self.time = self.timer.stop()

    def success(self, **results: Any) -> Trial.SuccessReport[Info]:
        """Generate a success report.

        ```python exec="true" source="material-block" result="python" title="success" hl_lines="7"
        from byop.optimization import Trial

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
        return Trial.SuccessReport(trial=self, time=time, results=results)

    def fail(self, **results: Any) -> Trial.FailReport[Info]:
        """Generate a failure report.

        ```python exec="true" source="material-block" result="python" title="fail" hl_lines="6 9 10"
        from byop.optimization import Trial

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
        self.results = results

        return Trial.FailReport(
            trial=self,
            time=time,
            exception=exception,
            results=results,
        )

    def crashed(
        self,
        exception: BaseException | None = None,
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
        assert exception is not None

        return Trial.CrashReport(trial=self, exception=exception)

    def store(
        self,
        items: Mapping[str, T],
        *,
        where: str | Path | Bucket | Callable[[str, Mapping[str, T]], None],
    ) -> None:
        """Store items related to the trial.

        ```python exec="true" source="material-block" result="python" title="store" hl_lines="5"
        from byop.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        print(trial.storage)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="store-bucket" hl_lines="8"
        from byop.optimization import Trial
        from byop.store import PathBucket

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
        self.storage = self.storage.union(items.keys())

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
        from byop.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        config = trial.retrieve("config.json", where="./results")
        print(config)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

        from byop.optimization import Trial
        from byop.store import PathBucket

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

    @dataclass
    class Report(ABC, Generic[InfoInner]):
        """A report for a trial.

        !!! note "Specific Instansiations"

            * [`SuccessReport`][byop.optimization.Trial.SuccessReport] -
                also contains a `.result` attribute which is where any
                results reported with [`success()`][byop.optimization.Trial.success]
                are stored.

            * [`FailReport`][byop.optimization.Trial.FailReport] -
                also contains a `.exception` attribute which is where any
                exceptions caught will be stored, along with `.results` where any
                results reported with [`fail()`][byop.optimization.Trial.fail] are
                stored.

            * [`CrashReport`][byop.optimization.Trial.CrashReport]

        Attributes:
            trial: The trial that was run.
        """

        successful: ClassVar[bool]
        status: ClassVar[str]

        trial: Trial[InfoInner]

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
        def stats(self) -> dict[str, Any]:
            """The stats of the trial."""
            return self.trial.stats

        @property
        def storage(self) -> set[str]:
            """The storage of the trial."""
            return self.trial.storage

        @property
        def info(self) -> InfoInner:
            """The info of the trial, specific to the optimizer that issued it."""
            return self.trial.info

        @abstractmethod
        def _extra_series(self) -> dict[str, Any]:
            """Additional information to put in the series."""

        def series(self) -> pd.Series:
            """Get a series of the results of the trial.

            !!! note "Prefixes"

                * `summary`: Entries will be prefixed with `#!python "summary:"`
                * `stats`: Entries will be prefixed with `#!python "stats:"`
                * `config`: Entries will be prefixed with `#!python "config:"`
                * `results`: Entries will be prefixed with `#!python "results:"`
            """
            return pd.Series(
                {
                    "name": self.name,
                    "successful": self.successful,
                    "status": self.status,
                    "trial_seed": self.trial.seed,
                    **self._extra_series(),
                    **prefix_keys(self.trial.summary, "summary:"),
                    **prefix_keys(self.trial.stats, "stats:"),
                    **prefix_keys(self.trial.config, "config:"),
                },
            )

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
            from byop.optimization import Trial

            trial = Trial(name="trial", config={"x": 1}, info={})

            trial.store({"config.json": trial.config}, where="./results")
            with trial.begin():
                report = trial.success()

            config = report.retrieve("config.json", where="./results")
            print(config)
            ```

            You could also create a Bucket and use that instead.

            ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

            from byop.optimization import Trial
            from byop.store import PathBucket

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

    @dataclass
    class CrashReport(Report[InfoInner]):
        """A report for a crashed trial.

        See [`Report`][byop.optimization.Trial.Report] for additional
        attributes.

        Attributes:
            exception: The exception for the trial.
        """

        successful: ClassVar[bool] = False
        status: ClassVar[Literal["crashed"]] = "crashed"

        exception: BaseException

        def _extra_series(self) -> dict[str, Any]:
            return {"exception": self.exception}

    @dataclass
    class SuccessReport(Report[InfoInner]):
        """A report for a successful trial.

        See [`Report`][byop.optimization.Trial.Report] for additional
        attributes.

        Attributes:
            time: The time taken by the trial.
            results: The results of the trial.
        """

        successful: ClassVar[bool] = True
        status: ClassVar[Literal["success"]] = "success"

        time: TimeInterval
        results: dict[str, Any] = field(default_factory=dict)

        def _extra_series(self) -> dict[str, Any]:
            return {
                **prefix_keys(self.results, "results:"),
                "time:start": self.time.start,
                "time:end": self.time.end,
                "time:duration": self.time.duration,
            }

    @dataclass
    class FailReport(Report[InfoInner]):
        """A report for a failed trial.

        Attributes:
            time: The time taken by the trial.
            exception: The exception raised by the trial if any.
            results: The results of the trial.
        """

        successful: ClassVar[bool] = False
        status: ClassVar[Literal["fail"]] = "fail"

        time: TimeInterval
        exception: BaseException | None
        results: dict[str, Any] = field(default_factory=dict)

        def _extra_series(self) -> dict[str, Any]:
            return {
                **prefix_keys(self.results, "results:"),
                "exception": self.exception,
                "time:start": self.time.start,
                "time:end": self.time.end,
                "time:duration": self.time.duration,
            }

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
        """A task that will run a target function and tell the optimizer the result."""

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
            call_limit: int | None = None,
            concurrent_limit: int | None = None,
            memory_limit: int | tuple[int, str] | None = None,
            cpu_time_limit: int | tuple[float, str] | None = None,
            wall_time_limit: int | tuple[float, str] | None = None,
        ) -> None:
            """Initialize a task.

            See [`Task`][byop.scheduling.task.Task] for more details.
            """
            super().__init__(
                function,
                scheduler,
                name=name,
                call_limit=call_limit,
                concurrent_limit=concurrent_limit,
                memory_limit=memory_limit,
                cpu_time_limit=cpu_time_limit,
                wall_time_limit=wall_time_limit,
            )
            self.trial_lookup: dict[Future, Trial] = {}

            self.on_f_returned(self._emit_report)
            self.on_f_exception(self._emit_report)

            self.on_report: Subscriber[Trial.Report[InfoInner]]
            self.on_report = self.subscriber(self.REPORT)

            self.on_failed: Subscriber[Trial.FailReport[InfoInner]]
            self.on_failed = self.subscriber(self.FAILURE)

            self.on_success: Subscriber[Trial.SuccessReport[InfoInner]]
            self.on_success = self.subscriber(self.SUCCESS)

            self.on_crashed: Subscriber[Trial.CrashReport[InfoInner]]
            self.on_crashed = self.subscriber(self.CRASHED)

        def __call__(
            self,
            trial: Trial[InfoInner],
        ) -> Future[Trial.Report[InfoInner]] | None:
            """Run the trial and return the future for the result.

            Args:
                trial: The trial to run.

            Returns:
                The future for the result of the trial.
            """
            future = super().__call__(trial)
            if future is not None:
                self.trial_lookup[future] = trial

            return future

        def _emit_report(
            self,
            future: Future,
            report: Trial.Report | BaseException,
        ) -> None:
            """Emit a report for a trial based on the type of the report."""
            # Emit the fact a report happened
            if isinstance(report, BaseException):
                report = self.trial_lookup[future].crashed(report)

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
