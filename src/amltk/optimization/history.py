"""The [`History`][amltk.optimization.History] is
used to keep a structured record of what occured with
[`Trial`][amltk.optimization.Trial]s and their associated
[`Report`][amltk.optimization.Trial.Report]s.

??? tip "Usage"

    ```python exec="true" source="material-block" html="true" hl_lines="19 23-24"
    from amltk.optimization import Trial, History, Metric
    from amltk.store import PathBucket

    loss = Metric("loss", minimize=True)

    def target_function(trial: Trial) -> Trial.Report:
        x = trial.config["x"]
        y = trial.config["y"]
        trial.store({"config.json": trial.config})

        with trial.begin():
            loss = x**2 - y

        if trial.exception:
            return trial.fail()

        return trial.success(loss=loss)

    # ... usually obtained from an optimizer
    bucket = PathBucket("all-trial-results")
    history = History()

    for x, y in zip([1, 2, 3], [4, 5, 6]):
        trial = Trial(name="some-unique-name", config={"x": x, "y": y}, bucket=bucket, metrics=[loss])
        report = target_function(trial)
        history.add(report)

    print(history.df())
    bucket.rmdir()  # markdon-exec: hide
    ```

You'll often need to perform some operations on a
[`History`][amltk.optimization.History] so we provide some utility functions here:

* [`filter(key=...)`][amltk.optimization.History.filter] - Filters the history by some
    predicate, e.g. `#!python history.filter(lambda report: report.status == "success")`
* [`groupby(key=...)`][amltk.optimization.History.groupby] - Groups the history by some
    key, e.g. `#!python history.groupby(lambda report: report.config["x"] < 5)`
* [`sortby(key=...)`][amltk.optimization.History.sortby] - Sorts the history by some
    key, e.g. `#!python history.sortby(lambda report: report.time.end)`

There is also some serialization capabilities built in, to allow you to store
your reports and load them back in later:

* [`df(...)`][amltk.optimization.History.df] - Output a `pd.DataFrame` of all
 the information available.
* [`from_df(...)`][amltk.optimization.History.from_df] - Create a `History` from
    a `pd.DataFrame`.

You can also retrieve individual reports from the history by using their
name, e.g. `#!python history["some-unique-name"]` or iterate through
the history with `#!python for report in history: ...`.
"""  # noqa: E501
from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar
from typing_extensions import override

import pandas as pd

from amltk._functional import compare_accumulate
from amltk._richutil import RichRenderable
from amltk.optimization.trial import Trial
from amltk.types import Comparable

if TYPE_CHECKING:
    from rich.console import RenderableType

    from amltk.optimization.metric import Metric

T = TypeVar("T")
CT = TypeVar("CT", bound=Comparable)
HashableT = TypeVar("HashableT", bound=Hashable)

# TODO: It might be faster to basically have the history
# always be a data frame, however this makes some things
# such as metrics a bit more difficult to work with.


@dataclass
class History(RichRenderable):
    """A history of trials.

    This is a collections of reports from trials, where you can access
    the reports by their trial name. It is unsorted in general, but
    by using [`sortby()`][amltk.optimization.History.sortby] you
    can sort the history.

    ```python exec="true" source="material-block" result="python" title="History"
    from amltk.optimization import Trial, History, Metric

    metric = Metric("cost", minimize=True)
    trials = [
        Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric])
        for i in range(10)
    ]
    history = History()

    for trial in trials:
        with trial.begin():
            x = trial.config["x"]
            report = trial.success(cost=x**2 - x*2 + 4)
            history.add(report)

    for report in history:
        print(f"{report.name=}, {report}")

    print(history.metrics)
    print(history.df())

    print(history.best())
    ```

    Attributes:
        reports: A mapping of trial names to reports.
    """

    reports: list[Trial.Report] = field(default_factory=list)
    metrics: dict[str, Metric] = field(default_factory=dict, repr=False)
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

    def best(self, metric: str | None = None) -> Trial.Report:
        """Returns the best report in the history.

        Args:
            metric: The metric to sort by. If `None`, it will use the
                first metric in the history. If there are multiple metrics
                and non are specified, it will raise an error.

        Returns:
            The best report.
        """
        if metric is None:
            if len(self.metrics) > 1:
                raise ValueError(
                    "There are multiple metrics in the history, "
                    "please specify which metric to sort by.",
                )

            _metric_def = next(iter(self.metrics.values()))
            _metric_name = _metric_def.name
        else:
            if metric not in self.metrics:
                raise ValueError(
                    f"Metric {metric} not found in history. "
                    f"Available metrics: {list(self.metrics.keys())}",
                )
            _metric_def = self.metrics[metric]
            _metric_name = metric

        _by = min if _metric_def.minimize else max
        return _by(self.reports, key=lambda r: r.metrics[_metric_name])

    def add(self, report: Trial.Report | Iterable[Trial.Report]) -> None:
        """Adds a report or reports to the history.

        Args:
            report: A report or reports to add.
        """
        match report:
            case Trial.Report():
                for m in report.metric_values:
                    if (_m := self.metrics.get(m.name)) is not None:
                        if m.metric != _m:
                            raise ValueError(
                                f"Metric {m.name} has conflicting definitions:"
                                f"\n{m.metric} != {_m}",
                            )
                    else:
                        self.metrics[m.name] = m.metric

                self.reports.append(report)
                self._lookup[report.name] = len(self.reports) - 1
            case reports:
                for _report in reports:
                    self.add(_report)

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
        metrics: bool = True,
        normalize_time: bool | float = True,
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame of the history.

        Each individual trial will be a row in the dataframe.

        !!! note "Prefixes"

            * `summary`: Entries will be prefixed with `#!python "summary:"`
            * `config`: Entries will be prefixed with `#!python "config:"`
            * `metrics`: Entries will be prefixed with `#!python "metrics:"`

        ```python exec="true" source="material-block" result="python" title="df" hl_lines="12"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
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
            metrics: Whether to include the metrics.
            normalize_time: Whether to normalize the time to the first
                report. If given a `#!python float`, it will normalize
                to that value.

                Will normalize all columns with `#!python "time:end"`.
                and `#!python "time:start"` in their name. It will use
                the time of the earliest report as the offset.

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
                    metrics=metrics,
                )
                for report in self.reports
            ],
        )
        _df = _df.convert_dtypes()

        match normalize_time:
            case True if "time:start" in _df.columns:
                time_columns = ("time:start", "time:end")
                cols = [c for c in _df.columns if c.endswith(time_columns)]
                _df[cols] -= _df["time:start"].min()
            case float():
                time_columns = ("time:start", "time:end")
                cols = [c for c in _df.columns if c.endswith(time_columns)]
                _df[cols] -= normalize_time
            case _:
                pass

        return _df

    def filter(self, key: Callable[[Trial.Report], bool]) -> History:
        """Filters the history by a predicate.

        ```python exec="true" source="material-block" result="python" title="filter" hl_lines="12"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        filtered_history = history.filter(lambda report: report.metrics["cost"] < 10)
        for report in filtered_history:
            cost = report.metrics["cost"]
            print(f"{report.name}, {cost=}, {report}")
        ```

        Args:
            key: A predicate to filter by.

        Returns:
            A new history with the filtered reports.
        """  # noqa: E501
        return History.from_reports([report for report in self.reports if key(report)])

    def groupby(
        self,
        key: Literal["status"] | Callable[[Trial.Report], Hashable],
    ) -> dict[Hashable, History]:
        """Groups the history by the values of a key.

        ```python exec="true" source="material-block" result="python" title="groupby" hl_lines="15"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
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

        You can pass a `#!python Callable` to group by any key you like:

        ```python exec="true" source="material-block" result="python"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.fail(cost=x)
                history.add(report)

        for below_5, history in history.groupby(lambda r: r.metrics["cost"] < 5).items():
            print(f"{below_5=}, {len(history)=}")
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

    def incumbents(
        self,
        key: Callable[[Trial.Report, Trial.Report], bool] | str,
        *,
        sortby: Callable[[Trial.Report], Comparable]
        | str = lambda report: report.time.end,
        reverse: bool | None = None,
        ffill: bool = False,
    ) -> list[Trial.Report]:
        """Returns a trace of the incumbents, where only the report that is better than the previous
        best report is kept.

        ```python exec="true" source="material-block" result="python" title="incumbents"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        incumbents = (
            history
            .incumbents("cost", sortby=lambda r: r.time.end)
        )
        for report in incumbents:
            print(f"{report.metrics=}, {report.config=}")
        ```

        Args:
            key: The key to use. If given a str, it will use that as the
                key to use in the metrics, defining if one report is better
                than another. If given a `#!python Callable`, it should
                return a `bool`, indicating if the first argument report
                is better than the second argument report.
            sortby: The key to sort by. If given a str, it will sort by
                the value of that key in the `.metrics` and also filter
                out anything that does not contain this key.
                By default, it will sort by the end time of the report.
            reverse: Whether to sort in some given order. By
                default (`None`), if given a metric key, the reports with
                the best metric values will be sorted first. If
                given a `#!python Callable`, the reports with the
                smallest values will be sorted first. Using
                `reverse=True` will always reverse this order, while
                `reverse=False` will always preserve it.
            ffill: Whether to forward fill the incumbents. This means that
                if a report is not an incumbent, it will be replaced with
                the current best. This is useful if you want to
                visualize the incumbents over some x axis, where the
                you have a point at every place along the axis.

        Returns:
            The history of incumbents.
        """  # noqa: E501
        match key:
            case str():
                metric = self.metrics[key]
                __op = operator.lt if metric.minimize else operator.gt  # type: ignore
                op = lambda r1, r2: __op(r1.metrics[key], r2.metrics[key])
            case _:
                op = key

        sorted_reports = self.sortby(sortby, reverse=reverse)
        return list(compare_accumulate(sorted_reports, op=op, ffill=ffill))

    def sortby(
        self,
        key: Callable[[Trial.Report], Comparable] | str,
        *,
        reverse: bool | None = None,
    ) -> list[Trial.Report]:
        """Sorts the history by a key and returns a sorted History.

        ```python exec="true" source="material-block" result="python" title="sortby" hl_lines="15"
        from amltk.optimization import Trial, History, Metric

        metric = Metric("cost", minimize=True)
        trials = [Trial(name=f"trial_{i}", config={"x": i}, metrics=[metric]) for i in range(10)]
        history = History()

        for trial in trials:
            with trial.begin():
                x = trial.config["x"]
                report = trial.success(cost=x**2 - x*2 + 4)
                history.add(report)

        trace = (
            history
            .filter(lambda report: report.status == "success")
            .sortby("cost")
        )

        for report in trace:
            print(f"{report.metrics}, {report}")
        ```

        Args:
            key: The key to sort by. If given a str, it will sort by
                the value of that key in the `.metrics` and also filter
                out anything that does not contain this key.
            reverse: Whether to sort in some given order. By
                default (`None`), if given a metric key, the reports with
                the best metric values will be sorted first. If
                given a `#!python Callable`, the reports with the
                smallest values will be sorted first. Using
                `reverse=True` will always reverse this order, while
                `reverse=False` will always preserve it.

        Returns:
            A sorted list of reports
        """  # noqa: E501
        # If given a str, filter out anything that doesn't have that key
        if isinstance(key, str):
            history = self.filter(lambda report: key in report.metric_names)
            sort_key: Callable[[Trial.Report], Comparable] = lambda r: r.metrics[key]
            reverse = (
                reverse if reverse is not None else (not self.metrics[key].minimize)
            )
        else:
            history = self
            sort_key = key
            reverse = False if reverse is None else reverse

        return sorted(history.reports, key=sort_key, reverse=reverse)

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
        return (
            self.reports == other.reports
            and self.metrics == other.metrics
            and self._lookup == other._lookup
        )

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

        return df_to_table(self.df(profiles=False), title="History", expand=True)

    @override
    def _repr_html_(self) -> str:
        return str(self.df().to_html())
