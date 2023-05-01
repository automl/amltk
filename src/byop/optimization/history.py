"""Classes for keeping track of trials."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import pandas as pd

from byop.optimization.trial import Trial
from byop.types import Comparable

T = TypeVar("T")
CT = TypeVar("CT", bound=Comparable)


@dataclass
class History(Mapping[str, Trial.Report]):
    """A history of trials.

    This is a collections of reports from trials, where you can access
    the reports by their trial name. It is unsorted in general, but
    by using [`sortby()`][byop.optimization.History.sortby] you
    can sort the history, giving you a [`Trace`][byop.optimization.Trace].

    It also provides some convenience methods, namely:

    * [`df()`][byop.optimization.history.History.df] to get a
        pandas DataFrame of the history.
    * [`filter()`][byop.optimization.history.History.filter] to
        filter the history by a predicate.

    ```python exec="true" source="material-block" result="python" title="History"
    from byop.optimization import Trial, History

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

        Each individual trial will be a row in the dataframe with the
        trial name as the index.

        !!! note "Prefixes"

            * `summary`: Entries will be prefixed with `#!python "summary:"`
            * `stats`: Entries will be prefixed with `#!python "stats:"`
            * `config`: Entries will be prefixed with `#!python "config:"`
            * `results`: Entries will be prefixed with `#!python "results:"`

        ```python exec="true" source="material-block" result="python" title="df" hl_lines="12"
        from byop.optimization import Trial, History

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
        series = [report.series() for report in self.reports.values()]
        history_df = pd.DataFrame(series)
        if len(history_df) > 0:
            history_df = history_df.set_index("name")
        else:
            history_df.index.name = "name"

        return history_df

    def filter(self, by: Callable[[Trial.Report], bool]) -> History:
        """Filters the history by a predicate.

        ```python exec="true" source="material-block" result="python" title="filter" hl_lines="12"
        from byop.optimization import Trial, History

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
        """  # noqa: E501
        return History(
            {name: report for name, report in self.reports.items() if by(report)},
        )

    def sortby(
        self,
        key: Callable[[Trial.Report], Comparable],
        *,
        reversed: bool = False,  # noqa: A002
    ) -> Trace:
        """Sorts the history by a key and returns a Trace.

        ```python exec="true" source="material-block" result="python" title="sortby" hl_lines="15"
        from byop.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.successful)
            .sortby(lambda report: report.time.end)
        )

        for report in trace:
            print(f"end={report.time.end}, {report}")
        ```

        Args:
            key: The key to sort by.
            reversed: Whether to sort in reverse order. By default,
                this is `False` meaning smaller items are sorted first.

        Returns:
            A Trace of the history.
        """  # noqa: E501
        return Trace(sorted(self.reports.values(), key=key, reverse=reversed))

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

    def save(self, path: str | Path | IO[str]) -> None:
        """Saves the history to a file.

        Args:
            path: The path to save the history to.
        """
        if isinstance(path, IO):
            path.write(self.df().to_csv())


@dataclass
class Trace(Sequence[Trial.Report]):
    """A trace of trials.

    A trace is a sequence of reports from trials that is ordered
    in some way.

    These are usually created using a [`History`][byop.optimization.History]
    object, specifically its [`sortby`][byop.optimization.History.sortby]
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

    def sortby(self, key: Callable[[Trial.Report], Any]) -> Trace:
        """Sorts the trace by a key.

        ```python exec="true" source="material-block" result="python" title="sortby" hl_lines="22"
        from byop.optimization import Trial, History

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
            key: A key to sort the trace by.

        Returns:
            A new trace with the sorted reports.
        """  # noqa: E501
        return Trace(sorted(self.reports, key=key))

    def filter(self, by: Callable[[Trial.Report], bool]) -> Trace:
        """Filters the trace by a predicate.

        ```python exec="true" source="material-block" result="python" title="filter" hl_lines="19"
        from byop.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.successful)
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
        key: Callable[[Trial.Report], CT],
        *,
        op: Callable[[CT, CT], bool] = operator.lt,
    ) -> IncumbentTrace:
        """Sorts the trace by a key and returns the incumbents.

        ```python exec="true" source="material-block" result="python" title="incumbents" hl_lines="22"
        from byop.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.successful)
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
            key: The key to use.
            op: The comparison operator to use when deciding if a report is
                an incumbent. By default, this is `operator.lt`, which means
                that the incumbent is the smallest value. If you want to maximize,
                you can use `operator.gt`. You can also use more advanced
                `Callable`s if you like.

        Returns:
            A Trace of the incumbents.
        """  # noqa: E501
        incumbents = [self.reports[0]]
        for report in self.reports[1:]:
            if op(key(report), key(incumbents[-1])):
                incumbents.append(report)

        return IncumbentTrace(incumbents)

    def df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of the trace.

        Each individual trial will be a row in the DataFrame with
        the index simply being their numerical order.

        !!! note "Prefixes"

            * `summary`: Entries will be prefixed with `#!python "summary:"`
            * `stats`: Entries will be prefixed with `#!python "stats:"`
            * `config`: Entries will be prefixed with `#!python "config:"`
            * `results`: Entries will be prefixed with `#!python "results:"`

        ```python exec="true" source="material-block" result="python" title="df" hl_lines="18"
        from byop.optimization import Trial, History

        trials = [Trial(f"trial_{i}", info=None, config={"x": i}) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.successful)
            .sortby(lambda report: report.time.end)
        )

        df = trace.df()
        print(df)
        ```

        Returns:
            A pandas DataFrame of the trace.
        """  # noqa: E501
        trace_df = pd.DataFrame([report.series() for report in self.reports])
        if len(trace_df) > 0:
            trace_df = trace_df.set_index("name")
        else:
            trace_df.index.name = "name"

        return trace_df


@dataclass
class IncumbentTrace(Trace):
    """A Trace of incumbents.

    Used primarily to distinguish between a general sorted
    [`Trace`][byop.optimization.Trace] and one with only the incumbents.
    """
