from __future__ import annotations

from more_itertools import pairwise
from pytest_cases import case, parametrize_with_cases

from byop.optimization import Trial
from byop.optimization.history import History, Trace


def quadratic(x):
    return x**2


@case
def case_empty() -> list[Trial.Report]:
    return []


@case(tags=["success"])
def case_one_report_success() -> list[Trial.Report]:
    trial = Trial("trial_1", info=None, config={"x": 1})
    with trial.begin():
        x = trial.config["x"]
        return [trial.success(cost=quadratic(x))]


@case(tags=["fail"])
def case_one_report_fail() -> list[Trial.Report]:
    trial = Trial("trial_1", info=None, config={"x": 1})
    with trial.begin():
        return [trial.fail(cost=100)]


@case(tags=["crash"])
def case_one_report_crash() -> list[Trial.Report]:
    trial = Trial("trial_1", info=None, config={"x": 1})
    return [trial.crashed(exception=ValueError())]


@case(tags=["success", "fail", "crash"])
def case_many_report() -> list[Trial.Report]:
    success_trials = [
        Trial(f"trial_{i+6}", info=None, config={"x": i}) for i in range(-5, 5)
    ]
    fail_trials = [
        Trial(f"trial_{i+16}", info=None, config={"x": i}) for i in range(-5, 5)
    ]
    crash_trials = [
        Trial(f"trial_{i+26}", info=None, config={"x": i}) for i in range(-5, 5)
    ]

    reports: list[Trial.Report] = []
    for trial in success_trials:
        with trial.begin():
            x = trial.config["x"]
            reports.append(trial.success(cost=quadratic(x)))

    for trial in fail_trials:
        with trial.begin():
            reports.append(trial.fail(cost=100))

    for trial in crash_trials:
        with trial.begin():
            x = trial.config["x"]
            reports.append(trial.crashed(exception=ValueError(x)))

    return reports


@parametrize_with_cases("reports", cases=".")
def test_history_add(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    assert len(history) == len(reports)


@parametrize_with_cases("reports", cases=".")
def test_history_df(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    history_df = history.df()
    assert len(history_df) == len(reports)
    assert history_df.index.name == "name"


@parametrize_with_cases("reports", cases=".")
def test_history_filter(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    history_df = history.df()
    filtered_success = history.filter(lambda report: report.status == "success")
    assert all(report.status == "success" for report in filtered_success.values())

    filtered_fail = history.filter(lambda report: report.status == "fail")
    assert all(report.status == "fail" for report in filtered_fail.values())

    filtered_crashed = history.filter(lambda report: report.status == "crashed")
    assert all(report.status == "crashed" for report in filtered_crashed.values())

    if len(history_df) > 0:
        counts = dict(history_df["status"].value_counts())
        assert counts.get("success", 0) == len(filtered_success)
        assert counts.get("fail", 0) == len(filtered_fail)
        assert counts.get("crashed", 0) == len(filtered_crashed)


@parametrize_with_cases("reports", cases=".")
def test_history_sortby_config(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_by_x_value = history.sortby(lambda report: report.config["x"])
    assert isinstance(trace_by_x_value, Trace)

    assert all(a.config["x"] <= b.config["x"] for a, b in pairwise(trace_by_x_value))


@parametrize_with_cases("reports", cases=".")
def test_trace_filter(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_by_x = history.sortby(lambda report: report.config["x"])
    trace_filtered = trace_by_x.filter(lambda report: report.config["x"] > 0)
    assert all(report.config["x"] > 0 for report in trace_filtered)


@parametrize_with_cases("reports", cases=".")
def test_trace_sortby(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_by_x = history.sortby(lambda report: report.config["x"])

    # Make sure that that it's sorted by the absolute value of x
    trace_sorted = trace_by_x.sortby(lambda report: abs(report.config["x"]))
    assert isinstance(trace_sorted, Trace)

    assert all(
        abs(a.config["x"]) <= abs(b.config["x"]) for a, b in pairwise(trace_sorted)
    )


@parametrize_with_cases("reports", cases=".")
def test_trace_df(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_by_x = history.sortby(lambda report: report.config["x"])
    trace_df = trace_by_x.df()

    assert len(trace_df) == len(trace_by_x) == len(history)
    assert trace_df.index.name == "name"

    filtered_success = trace_by_x.filter(lambda report: report.status == "success")
    assert all(report.status == "success" for report in filtered_success)

    filtered_fail = trace_by_x.filter(lambda report: report.status == "fail")
    assert all(report.status == "fail" for report in filtered_fail)

    filtered_crashed = trace_by_x.filter(lambda report: report.status == "crashed")
    assert all(report.status == "crashed" for report in filtered_crashed)

    if len(trace_df) > 0:
        counts = dict(trace_df["status"].value_counts())
        assert counts.get("success", 0) == len(filtered_success)
        assert counts.get("fail", 0) == len(filtered_fail)
        assert counts.get("crashed", 0) == len(filtered_crashed)
