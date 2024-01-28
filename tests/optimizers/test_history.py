from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
from more_itertools import pairwise
from pytest_cases import case, parametrize_with_cases

from amltk.optimization import History, Metric, Trial

metrics = {"loss": Metric("loss", minimize=True)}


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
        with _trial.profile("trial"):
            with _trial.profile("dummy"):
                pass

            if fail is not None:
                reports.append(_trial.fail(loss=fail))
            else:
                x = _trial.config["x"]
                reports.append(_trial.success(loss=quadratic(x)))

    return reports


@case
def case_empty() -> list[Trial.Report]:
    return []


@case(tags=["success"])
def case_one_report_success() -> list[Trial.Report]:
    trial: Trial = Trial.create(name="trial_1", config={"x": 1}, metrics=metrics)
    return eval_trial(trial)


@case(tags=["fail"])
def case_one_report_fail() -> list[Trial.Report]:
    trial: Trial = Trial.create(name="trial_1", config={"x": 1}, metrics=metrics)
    return eval_trial(trial, fail=100)


@case(tags=["crash"])
def case_one_report_crash() -> list[Trial.Report]:
    trial: Trial = Trial.create(name="trial_1", config={"x": 1}, metrics=metrics)
    return eval_trial(trial, crash=ValueError("Some Error"))


@case(tags=["success", "fail", "crash"])
def case_many_report() -> list[Trial.Report]:
    success_trials: list[Trial] = [
        Trial.create(name=f"trial_{i+6}", config={"x": i}, metrics=metrics)
        for i in range(-5, 5)
    ]
    fail_trials: list[Trial] = [
        Trial.create(name=f"trial_{i+16}", config={"x": i}, metrics=metrics)
        for i in range(-5, 5)
    ]
    crash_trials: list[Trial] = [
        Trial.create(name=f"trial_{i+26}", config={"x": i}, metrics=metrics)
        for i in range(-5, 5)
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
    assert all(report.status == "success" for report in filtered_success)

    filtered_fail = history.filter(lambda report: report.status == "fail")
    assert all(report.status == "fail" for report in filtered_fail)

    filtered_crashed = history.filter(lambda report: report.status == "crashed")
    assert all(report.status == "crashed" for report in filtered_crashed)

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
    assert all(a.config["x"] <= b.config["x"] for a, b in pairwise(trace_by_x_value))


@parametrize_with_cases("reports", cases=".")
def test_trace_filter(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_by_x = history.filter(lambda report: report.config["x"] > 0).sortby(
        lambda report: report.config["x"],
    )
    assert all(report.config["x"] > 0 for report in trace_by_x)


@parametrize_with_cases("reports", cases=".")
def test_trace_sortby(reports: list[Trial.Report]) -> None:
    history = History()
    history.add(reports)

    trace_sorted = history.sortby(lambda report: abs(report.config["x"]))

    assert all(
        abs(a.config["x"]) <= abs(b.config["x"]) for a, b in pairwise(trace_sorted)
    )


def test_history_sortby() -> None:
    trials: list[Trial] = [
        Trial.create(name=f"trial_{i+6}", metrics=metrics, config={"x": i})
        for i in range(-5, 5)
    ]

    summary_items = ["trial_1", "trial_3"]
    history = History()

    for trial in trials:
        if trial.name in summary_items:
            trial.summary["other_loss"] = trial.config["x"] ** 2

        report = trial.success(loss=trial.config["x"])
        history.add(report)

    trace_loss = history.sortby("loss")
    assert len(trace_loss) == len(trials)
    losses = [r.values["loss"] for r in trace_loss]
    assert sorted(losses) == losses

    trace_other = history.filter(lambda report: "other_loss" in report.summary).sortby(
        lambda report: report.summary["other_loss"],
    )
    assert len(trace_other) == len(summary_items)
    assert all("other_loss" in r.summary for r in trace_other)

    losses = [r.summary["other_loss"] for r in trace_other]
    assert sorted(losses) == losses


def test_history_incumbents() -> None:
    m1 = Metric("score", minimize=False)
    m2 = Metric("loss", minimize=True)
    trials: list[Trial] = [
        Trial.create(
            name=f"trial_{i+6}",
            metrics={"score": m1, "loss": m2},
            config={"x": i},
        )
        for i in [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]
    ]
    history = History()

    for trial in trials:
        x = trial.config["x"]
        report = trial.success(loss=x, score=x)
        history.add(report)

    hist_1 = history.incumbents("loss", ffill=True)
    expected_1 = [0, -1, -1, -3, -3, -5, -5, -7, -7, -9]
    assert [r.values["loss"] for r in hist_1] == expected_1

    hist_2 = history.incumbents("loss", ffill=False)
    expected_2 = [0, -1, -3, -5, -7, -9]
    assert [r.values["loss"] for r in hist_2] == expected_2

    hist_3 = history.incumbents("score", ffill=True)
    expected_3 = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
    assert [r.values["score"] for r in hist_3] == expected_3

    hist_4 = history.incumbents("score", ffill=False)
    expected_4 = [0, 2, 4, 6, 8]
    assert [r.values["score"] for r in hist_4] == expected_4


@parametrize_with_cases("reports", cases=".")
def test_history_serialization(reports: list[Trial.Report], tmp_path: Path) -> None:
    history = History()
    history.add(reports)

    if any(reports):
        report_df = reports[0].df()
        restored_report_df = Trial.Report.from_df(report_df).df()
        pd.testing.assert_frame_equal(report_df, restored_report_df)

    df = history.df(normalize_time=False)
    assert len(df) == len(reports)

    restored_history = History.from_df(df)
    restored_df = restored_history.df(normalize_time=False)

    pd.testing.assert_frame_equal(df, restored_df)
    pd.set_option("display.precision", 8)

    tmpfile = tmp_path / "history.csv"
    history.df(normalize_time=False).to_csv(tmpfile)

    restored_history = History.from_df(
        pd.read_csv(tmpfile, float_precision="round_trip"),  # type: ignore
    )
    restored_df = restored_history.df(normalize_time=False)
    pd.testing.assert_frame_equal(df, restored_df, atol=1e-9)
