"""The [`History`][amltk.optimization.History] is
used to keep a structured record of what occured with
[`Trial`][amltk.optimization.Trial]s and their associated
[`Report`][amltk.optimization.Trial.Report]s.

??? tip "Usage"

    ```python exec="true" source="material-block" html="true" hl_lines="19 23-24"
    from amltk.optimization import Trial, History
    from amltk.store import PathBucket

    def target_function(trial: Trial) -> Trial.Report:
        x = trial.config["x"]
        y = trial.config["y"]
        trial.store({"config.json": trial.config})

        with trial.begin():
            cost = x**2 - y

        if trial.exception:
            return trial.fail()

        return trial.success(cost=cost)

    # ... usually obtained from an optimizer
    bucket = PathBucket("all-trial-results")
    history = History()

    for x, y in zip([1, 2, 3], [4, 5, 6]):
        trial = Trial(name="some-unique-name", config={"x": x, "y": y}, bucket=bucket)
        report = target_function(trial)
        history.add(report)

    print(history.df())
    bucket.rmdir()  # markdon-exec: hide
    ```

You'll often need to perform some operations on a
[`History`][amltk.optimization.History] so we provide some utility functions here:

* [`filter(by=...)`][amltk.optimization.History.filter] - Filters the history by some
    predicate, e.g. `#!python history.filter(lambda report: report.status == "success")`
* [`groupby(key=...)`][amltk.optimization.History.groupby] - Groups the history by some
    key, e.g. `#!python history.groupby(lambda report: report.config["x"] < 5)`
* [`sortby(key=...)`][amltk.optimization.History.sortby] - Sorts the history by some
    key, e.g. `#!python history.sortby(lambda report: report.time.end)`

    This will return a [`Trace`][amltk.optimization.Trace] which is the same
    as a `History` in many respects, other than the fact it now has a sorted order.

There is also some serialization capabilities built in, to allow you to store
your results and load them back in later:

* [`df(...)`][amltk.optimization.History.df] - Output a `pd.DataFrame` of all
 the information available.
* [`from_df(...)`][amltk.optimization.History.from_df] - Create a `History` from
    a `pd.DataFrame`.

You can also retrieve individual reports from the history by using their
name, e.g. `#!python history["some-unique-name"]` or iterate through
the history with `#!python for report in history: ...`.
"""
from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Literal,
    TypeVar,
    overload,
)
from typing_extensions import override

import pandas as pd

from amltk._functional import compare_accumulate
from amltk._richutil import RichRenderable
from amltk.optimization.trial import Trial
from amltk.types import Comparable

if TYPE_CHECKING:
    from rich.console import RenderableType

T = TypeVar("T")
CT = TypeVar("CT", bound=Comparable)
HashableT = TypeVar("HashableT", bound=Hashable)


@dataclass
class History(RichRenderable):
    """A history of trials.

    This is a collections of reports from trials, where you can access
    the reports by their trial name. It is unsorted in general, but
    by using [`sortby()`][amltk.optimization.History.sortby] you
    can sort the history, giving you a [`Trace`][amltk.optimization.Trace].

    It also provides some convenience methods, namely:

    * [`df()`][amltk.optimization.history.History.df] to get a
        pandas DataFrame of the history.
    * [`filter()`][amltk.optimization.history.History.filter] to
        filter the history by a predicate.

    ```python exec="true" source="material-block" result="python" title="History"
    from amltk.optimization import Trial, History

    trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
    history = History()

    for trial in trials:
        with trial.begin():
            x = trial.config["x"]
            report = trial.success(cost=x**2 - x*2 + 4)
            history.add(report)

    for report in history:
        print(f"{report.name=}, {report}")

    print(history.df())
    ```

    Attributes:
        reports: A mapping of trial names to reports.
    """

    reports: list[Trial.Report] = field(default_factory=list)
    _lookup: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_reports(cls, reports: Iterable[Trial.Report]) -> History:
        """Creates a history from reports.

        Args:
            reports: An iterable of reports.

        Returns:
            A history.
        """
        history = cls()
        history.add(reports)
        return history

    def add(self, report: Trial.Report | Iterable[Trial.Report]) -> None:
        """Adds a report or reports to the history.

        Args:
            report: A report or reports to add.
        """
        if isinstance(report, Trial.Report):
            self.reports.append(report)
            self._lookup[report.name] = len(self.reports) - 1
            return

        for _report in report:
            self.reports.append(_report)
            self._lookup[_report.name] = len(self.reports) - 1

    def find(self, name: str) -> Trial.Report:
        """Finds a report by trial name.

        Args:
            name: The name of the trial.

        Returns:
            The report.
        """
        return self.reports[self._lookup[name]]

    def df(
        self,
        *,
        profiles: bool = True,
        configs: bool = True,
        summary: bool = True,
        results: bool = True,
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame of the history.

        Each individual trial will be a row in the dataframe.

        !!! note "Prefixes"

            * `summary`: Entries will be prefixed with `#!python "summary:"`
            * `config`: Entries will be prefixed with `#!python "config:"`
            * `results`: Entries will be prefixed with `#!python "results:"`

        ```python exec="true" source="material-block" result="python" title="df" hl_lines="12"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        print(history.df())
        ```

        Args:
            profiles: Whether to include the profiles.
            configs: Whether to include the configs.
            summary: Whether to include the summary.
            results: Whether to include the results.

        Returns:
            A pandas DataFrame of the history.
        """  # noqa: E501
        if len(self) == 0:
            return pd.DataFrame()

        _df = pd.concat(
            [
                report.df(
                    profiles=profiles,
                    configs=configs,
                    summary=summary,
                    results=results,
                )
                for report in self.reports
            ],
        )
        return _df.convert_dtypes()

    def filter(self, by: Callable[[Trial.Report], bool]) -> History:
        """Filters the history by a predicate.

        ```python exec="true" source="material-block" result="python" title="filter" hl_lines="12"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        filtered_history = history.filter(lambda report: report.results["cost"] < 10)
        for report in filtered_history:
            cost = report.results["cost"]
            print(f"{report.name}, {cost=}, {report}")
        ```

        Args:
            by: A predicate to filter by.

        Returns:
            A new history with the filtered reports.
        """  # noqa: E501
        return History.from_reports([report for report in self.reports if by(report)])

    def groupby(
        self,
        key: Literal["status"] | Callable[[Trial.Report], Hashable],
    ) -> dict[Hashable, History]:
        """Groups the history by the values of a key.

        ```python exec="true" source="material-block" result="python" title="groupby" hl_lines="15"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                if x % 2 == 0:
                    report = trial.fail(cost=1_000)
                else:
                    report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        for status, history in history.groupby("status").items():
            print(f"{status=}, {len(history)=}")
        ```

        Args:
            key: A key to group by. If `"status"` is passed, the history will be
                grouped by the status of the reports.

        Returns:
            A mapping of keys to histories.
        """  # noqa: E501
        d = defaultdict(list)

        if key == "status":
            key = operator.attrgetter("status")

        for report in self.reports:
            d[key(report)].append(report)

        return {k: History.from_reports(v) for k, v in d.items()}

    def sortby(
        self,
        key: Callable[[Trial.Report], Comparable] | str,
        *,
        reverse: bool = False,
    ) -> Trace:
        """Sorts the history by a key and returns a Trace.

        ```python exec="true" source="material-block" result="python" title="sortby" hl_lines="15"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.status == "success")
            .sortby(lambda report: report.time.end)
        )

        for report in trace:
            print(f"end={report.time.end}, {report}")
        ```

        Args:
            key: The key to sort by. If given a str, it will sort by
                the value of that key in the summary and also filter
                out anything that does not contain this key.
            reverse: Whether to sort in reverse order. By default,
                this is `False` meaning smaller items are sorted first.

        Returns:
            A Trace of the history.
        """  # noqa: E501
        # If given a str, filter out anything that doesn't have that key
        if isinstance(key, str):
            history = self.filter(lambda report: key in report.summary)
            sort_key = lambda report: report.summary[key]
        else:
            history = self
            sort_key = key

        return Trace(sorted(history.reports, key=sort_key, reverse=reverse))

    @override
    def __getitem__(  # type: ignore
        self,
        key: int | str | slice,
    ) -> Trial.Report | History:
        if isinstance(key, str):
            return self.find(key)
        if isinstance(key, int):
            return self.reports[key]

        return History.from_reports(self.reports[key])

    def __iter__(self) -> Iterator[Trial.Report]:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, History):
            return NotImplemented
        return self.reports == other.reports

    def to_csv(self, path: str | Path | IO[str]) -> None:
        """Saves the history to a csv.

        Args:
            path: The path to save the history to.
        """
        if isinstance(path, IO):
            path.write(self.df().to_csv(na_rep=""))
        else:
            self.df().to_csv(path)

    @classmethod
    def from_csv(cls, path: str | Path | IO[str] | pd.DataFrame) -> History:
        """Loads a history from a csv.

        Args:
            path: The path to load the history from.

        Returns:
            A History.
        """
        _df = (
            pd.read_csv(
                path,  # type: ignore
                float_precision="round_trip",  # type: ignore
            )
            if isinstance(path, IO | str | Path)
            else path
        )

        return cls.from_df(_df.convert_dtypes())

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> History:
        """Loads a history from a pandas DataFrame.

        Args:
            df: The DataFrame to load the history from.

        Returns:
            A History.
        """
        if len(df) == 0:
            return cls()
        return History.from_reports(Trial.Report.from_df(s) for _, s in df.iterrows())

    @override
    def __rich__(self) -> RenderableType:
        from amltk._richutil import df_to_table

        return df_to_table(
            self.df(configs=False, profiles=False, summary=False),
            title="History",
            expand=True,
        )

    @override
    def _repr_html_(self) -> str:
        return str(self.df().to_html())


@dataclass
class Trace(Sequence[Trial.Report]):
    """A trace of trials.

    A trace is a sequence of reports from trials that is ordered
    in some way.

    These are usually created using a [`History`][amltk.optimization.History]
    object, specifically its [`sortby`][amltk.optimization.History.sortby]
    method to create the order.

    Attributes:
        reports: The reports in the trace.
    """

    reports: list[Trial.Report]

    @overload
    def __getitem__(self, key: int) -> Trial.Report:
        ...

    @overload
    def __getitem__(self, key: slice) -> Trace:
        ...

    @override
    def __getitem__(self, key: int | slice) -> Trial.Report | Trace:
        if isinstance(key, int):
            return self.reports[key]
        return Trace(self.reports[key])

    @override
    def __iter__(self) -> Iterator[Trial.Report]:
        return iter(self.reports)

    @override
    def __len__(self) -> int:
        return len(self.reports)

    @override
    def __repr__(self) -> str:
        return f"Trace({self.reports})"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trace):
            return NotImplemented
        return self.reports == other.reports

    def sortby(self, key: Callable[[Trial.Report], Comparable] | str) -> Trace:
        """Sorts the trace by a key.

        ```python exec="true" source="material-block" result="python" title="sortby" hl_lines="22"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        # Sortby the x values
        trace = history.sortby(lambda report: report.trial.config["x"])

        print("--history sorted by x--")
        for report in trace:
            x = report.trial.config["x"]
            cost = report.results["cost"]
            print(f"{x=}, {cost=}, {report}")

        # Sort the trace by the cost
        trace = trace.sortby(lambda report: report.results["cost"])

        print("--trace sorted by cost--")
        for report in trace:
            x = report.trial.config["x"]
            cost = report.results["cost"]
            print(f"{x=}, {cost=}, {report}")
        ```

        Args:
            key: A key to sort the trace by. If given a str, it will
                sort by the value of that key in the summary and also
                filter out anything that does not contain this key in
                its summary.

        Returns:
            A new trace with the sorted reports.
        """  # noqa: E501
        if isinstance(key, str):
            trace = self.filter(lambda report: key in report.summary)
            sort_key = lambda report: report.summary[key]
        else:
            trace = self
            sort_key = key

        return Trace(sorted(trace.reports, key=sort_key))

    def filter(self, by: Callable[[Trial.Report], bool]) -> Trace:
        """Filters the trace by a predicate.

        ```python exec="true" source="material-block" result="python" title="filter" hl_lines="19"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.status == "success")
            .sortby(lambda report: report.time.end)
        )
        print(f"Length pre-filter: {len(trace)}")

        filtered = trace.filter(lambda report: report.results["cost"] < 10)

        print(f"Length post-filter: {len(filtered)}")
        for report in filtered:
            cost = report.results["cost"]
            print(f"{cost=}, {report}")
        ```

        Args:
            by: A predicate to filter the trace by.

        Returns:
            A new trace with the filtered reports.
        """  # noqa: E501
        return Trace([report for report in self.reports if by(report)])

    def incumbents(
        self,
        key: Callable[[Trial.Report], CT] | str,
        *,
        op: Callable[[CT, CT], bool] | Literal["min"] | Literal["max"] = "min",
        ffill: bool = False,
    ) -> IncumbentTrace:
        """Sorts the trace by a key and returns the incumbents.

        ```python exec="true" source="material-block" result="python" title="incumbents" hl_lines="22"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.status == "success")
            .sortby(lambda report: report.time.end)
        )
        print("--trace--")
        for report in trace:
            cost = report.results["cost"]
            print(f"{cost=}, {report}")

        incumbents = trace.incumbents(lambda report: report.results["cost"])
        print("--incumbents--")
        for report in incumbents:
            print(report)
            cost = report.results["cost"]
            print(f"{cost=}, {report}")
        ```

        Args:
            key: The key to use. If given a str, it will sort by the value
                of that key in the summary and also filter out anything that
                does not contain this key in its summary.
            op: The comparison operator to use when deciding if a report is
                an incumbent. By default, this is `"min"`, which means
                that the incumbent is the smallest value. If you want to maximize,
                you can use `"max"`. You can also use more advanced
                `Callable`s if you like.
            ffill: Whether to forward fill the incumbents. This means that
                if a report is not an incumbent, it will be replaced with
                the previous incumbent. This is useful if you want to
                visualize the incumbents over time.

        Returns:
            A Trace of the incumbents.
        """  # noqa: E501
        if isinstance(op, str):
            if op not in {"min", "max"}:
                raise ValueError(f"Unknown op: {op}")
            op = operator.lt if op == "min" else operator.gt  # type: ignore

        if isinstance(key, str):
            trace = self.filter(lambda report: key in report.summary)
            _op = lambda r1, r2: op(r1.summary[key], r2.summary[key])  # type: ignore
        else:
            trace = self
            _op = lambda r1, r2: op(key(r1), key(r2))  # type: ignore

        incumbents = list(
            compare_accumulate(trace.reports, op=_op, ffill=ffill),  # type: ignore
        )
        return IncumbentTrace(incumbents)

    def df(
        self,
        *,
        profiles: bool = True,
        configs: bool = True,
        summary: bool = True,
        results: bool = True,
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame of the trace.

        Each individual trial will be a row in the DataFrame

        !!! note "Prefixes"

            * `summary`: Entries will be prefixed with `#!python "summary:"`
            * `config`: Entries will be prefixed with `#!python "config:"`
            * `results`: Entries will be prefixed with `#!python "results:"`

        ```python exec="true" source="material-block" result="python" title="df" hl_lines="18"
        from amltk.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.status == "success")
            .sortby(lambda report: report.time.end)
        )

        df = trace.df()
        print(df)
        ```

        Args:
            profiles: Whether to include the profiles.
            configs: Whether to include the configs.
            summary: Whether to include the summary.
            results: Whether to include the results.

        Returns:
            A pandas DataFrame of the trace.
        """  # noqa: E501
        if len(self) == 0:
            return pd.DataFrame()

        return pd.concat(
            [
                report.df(
                    profiles=profiles,
                    configs=configs,
                    summary=summary,
                    results=results,
                )
                for report in self.reports
            ],
            ignore_index=True,
        ).convert_dtypes()

    def to_csv(self, path: str | Path | IO[str]) -> None:
        """Saves the history to a csv.

        Args:
            path: The path to save the history to.
        """
        if isinstance(path, IO):
            path.write(self.df().to_csv())
        else:
            self.df().to_csv(path)

    def _ipython_display(self) -> None:
        from IPython.display import display

        display(self.df(profiles=False))


@dataclass
class IncumbentTrace(Trace):
    """A Trace of incumbents.

    Used primarily to distinguish between a general sorted
    [`Trace`][amltk.optimization.Trace] and one with only the incumbents.
    """
