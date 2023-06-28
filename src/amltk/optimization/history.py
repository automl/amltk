"""Classes for keeping track of trials."""
from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    IO,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import pandas as pd

from amltk.functional import compare_accumulate
from amltk.optimization.trial import Trial
from amltk.types import Comparable

T = TypeVar("T")
CT = TypeVar("CT", bound=Comparable)
HashableT = TypeVar("HashableT", bound=Hashable)


@dataclass
class History(Mapping[str, Trial.Report]):
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

    for name, report in history.items():
        print(f"{name=}, {report}")

    print(history.df())
    ```

    Attributes:
        reports: A mapping of trial names to reports.
    """

    reports: dict[str, Trial.Report] = field(default_factory=dict)

    @classmethod
    def from_reports(cls, reports: Iterable[Trial.Report]) -> History:
        """Creates a history from reports.

        Args:
            reports: An iterable of reports.

        Returns:
            A history.
        """
        return cls({report.name: report for report in reports})

    def add(self, report: Trial.Report | Iterable[Trial.Report]) -> None:
        """Adds a report or reports to the history.

        Args:
            report: A report or reports to add.
        """
        if isinstance(report, Trial.Report):
            self.reports[report.name] = report
        else:
            self.reports.update({r.name: r for r in report})

    def df(self) -> pd.DataFrame:
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

        Returns:
            A pandas DataFrame of the history.
        """  # noqa: E501
        if len(self) == 0:
            return pd.DataFrame()

        history_df = pd.concat([report.df() for report in self.reports.values()])
        if (
            len(history_df) > 0
            and "time:start" in history_df.columns
            and "time:end" in history_df.columns
        ):
            min_time = history_df[["time:start", "time:end"]].min().min()
            history_df["time:relative_start"] = history_df["time:start"] - min_time
            history_df["time:relative_end"] = history_df["time:end"] - min_time

        return history_df

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
        for name, report in filtered_history.items():
            cost = report.results["cost"]
            print(f"{name}, {cost=}, {report}")
        ```

        Args:
            by: A predicate to filter by.

        Returns:
            A new history with the filtered reports.
        """  # noqa: E501
        return History(
            {name: report for name, report in self.reports.items() if by(report)},
        )

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

        for report in self.reports.values():
            d[key(report)].append(report)

        return {k: History.from_reports(v) for k, v in d.items()}

    def sortby(
        self,
        key: Callable[[Trial.Report], Comparable] | str,
        *,
        reverse: bool = False,  # noqa: A002
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

        return Trace(sorted(history.reports.values(), key=sort_key, reverse=reverse))

    def __getitem__(self, key: str) -> Trial.Report:
        return self.reports[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    def __repr__(self) -> str:
        return f"History({self.reports})"

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
        _df = pd.read_csv(path) if isinstance(path, (IO, str, Path)) else path

        present_cols = {
            k: v for k, v in Trial.Report.DF_COLUMN_TYPES.items() if k in _df
        }
        _df = _df.astype(present_cols)

        return cls.from_df(_df)

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
        reports = [Trial.Report.from_df(s) for _, s in df.iterrows()]
        return History({report.name: report for report in reports})


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

    def __getitem__(self, key: int | slice) -> Trial.Report | Trace:
        if isinstance(key, int):
            return self.reports[key]
        return Trace(self.reports[key])

    def __iter__(self) -> Iterator[Trial.Report]:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    def __repr__(self) -> str:
        return f"Trace({self.reports})"

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
        else:
            op = op  # type: ignore

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

    def df(self) -> pd.DataFrame:
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

        Returns:
            A pandas DataFrame of the trace.
        """  # noqa: E501
        if len(self) == 0:
            return pd.DataFrame()

        trace_df = pd.concat([report.df() for report in self.reports])
        if (
            len(trace_df) > 0
            and "time:start" in trace_df.columns
            and "time:end" in trace_df.columns
        ):
            min_time = trace_df[["time:start", "time:end"]].min().min()
            trace_df["time:relative_start"] = trace_df["time:start"] - min_time
            trace_df["time:relative_end"] = trace_df["time:end"] - min_time

        return trace_df

    def to_csv(self, path: str | Path | IO[str]) -> None:
        """Saves the history to a csv.

        Args:
            path: The path to save the history to.
        """
        if isinstance(path, IO):
            path.write(self.df().to_csv())
        else:
            self.df().to_csv(path)


@dataclass
class IncumbentTrace(Trace):
    """A Trace of incumbents.

    Used primarily to distinguish between a general sorted
    [`Trace`][amltk.optimization.Trace] and one with only the incumbents.
    """
