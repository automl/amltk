from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
from more_itertools import pairwise
from pytest_cases import case, parametrize_with_cases

from amltk.optimization import Trial
from amltk.optimization.history import History, Trace


def quadratic(x: float) -> float:
    return x**2


def eval_trial(
    trial: Trial | list[Trial],
    *,
    fail: float | None = None,
    crash: Exception | None = None,
) -> list[Trial.Report]:
    if isinstance(trial, Trial):
        trial = [trial]

    if crash is not None:
        try:
            raise crash
        except Exception as crash:  # noqa: BLE001
            return [
                trial.crashed(exception=crash, traceback=traceback.format_exc())
                for trial in trial
            ]

    reports: list[Trial.Report] = []
    for _trial in trial:
        with _trial.begin():
            with _trial.profile("dummy"):
                pass

            if fail is not None:
                reports.append(_trial.fail(cost=fail))
            else:
                x = _trial.config["x"]
                reports.append(_trial.success(cost=quadratic(x)))

    return reports


@case
def case_empty() -> list[Trial.Report]:
    return []


@case(tags=["success"])
def case_one_report_success() -> list[Trial.Report]:
    trial: Trial = Trial("trial_1", info=None, config={"x": 1})
    return eval_trial(trial)


@case(tags=["fail"])
def case_one_report_fail() -> list[Trial.Report]:
    trial: Trial = Trial("trial_1", info=None, config={"x": 1})
    return eval_trial(trial, fail=100)


@case(tags=["crash"])
def case_one_report_crash() -> list[Trial.Report]:
    trial: Trial = Trial("trial_1", info=None, config={"x": 1})
    return eval_trial(trial, crash=ValueError("Some Error"))


@case(tags=["success", "fail", "crash"])
def case_many_report() -> list[Trial.Report]:
    success_trials: list[Trial] = [
        Trial(f"trial_{i+6}", info=None, config={"x": i}) for i in range(-5, 5)
    ]
    fail_trials: list[Trial] = [
        Trial(f"trial_{i+16}", info=None, config={"x": i}) for i in range(-5, 5)
    ]
    crash_trials: list[Trial] = [
        Trial(f"trial_{i+26}", info=None, config={"x": i}) for i in range(-5, 5)
    ]

    return [
        *eval_trial(success_trials),
        *eval_trial(fail_trials, fail=100),
        *eval_trial(crash_trials, crash=ValueError("Crash Error")),
    ]


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


def test_history_sortby() -> None:
    trials: list[Trial] = [
        Trial(f"trial_{i+6}", info=None, config={"x": i}) for i in range(-5, 5)
    ]

    summary = ["trial_1", "trial_3"]
    history = History()

    for trial in trials:
        with trial.begin():
            if trial.name in summary:
                trial.summary["loss"] = trial.config["x"] ** 2

            report = trial.success(cost=trial.config["x"])
            history.add(report)

    trace = history.sortby("loss")
    assert isinstance(trace, Trace)
    assert len(trace) == len(summary)
    assert all("loss" in r.summary for r in trace)

    losses = [r.summary["loss"] for r in trace]
    assert sorted(losses) == losses


@parametrize_with_cases("reports", cases=".")
def test_history_serialization(reports: list[Trial.Report], tmp_path: Path) -> None:
    history = History()
    history.add(reports)

    if any(reports):
        report_df = reports[0].df()
        restored_report_df = Trial.Report.from_df(report_df).df()
        pd.testing.assert_frame_equal(report_df, restored_report_df)

    df = history.df()
    assert len(df) == len(reports)

    restored_history = History.from_df(df)
    restored_df = restored_history.df()

    pd.testing.assert_frame_equal(df, restored_df)
    pd.set_option("display.precision", 8)

    tmpfile = tmp_path / "history.csv"
    history.to_csv(tmpfile)

    restored_history = History.from_csv(tmpfile)
    restored_df = restored_history.df()
    pd.testing.assert_frame_equal(df, restored_df, atol=1e-9)
