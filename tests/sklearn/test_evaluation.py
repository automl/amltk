from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pytest_cases import case, parametrize, parametrize_with_cases
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amltk.optimization.trial import Metric, Trial
from amltk.pipeline import Component, request
from amltk.sklearn.evaluation import (
    CVEvaluation,
    ImplicitMetricConversionWarning,
    TaskTypeName,
    TaskTypeWarning,
    _default_cv_resampler,
    _default_holdout,
    identify_task_type,
)


# NOTE: We can cache this as it doesn't get changed
@cache
def data_for_task_type(task_type: TaskTypeName) -> tuple[np.ndarray, np.ndarray]:
    match task_type:
        case "binary":
            return make_classification(
                random_state=42,
                n_samples=20,
                n_classes=2,
                n_informative=3,
            )  # type: ignore
        case "multiclass":
            return make_classification(
                random_state=42,
                n_samples=20,
                n_classes=4,
                n_informative=3,
            )  # type: ignore
        case "multilabel-indicator":
            x, y = make_classification(
                random_state=42,
                n_samples=20,
                n_classes=2,
                n_informative=3,
            )
            y = np.vstack([y, y]).T
            return x, y  # type: ignore
        case "multiclass-multioutput":
            x, y = make_classification(
                random_state=42,
                n_samples=20,
                n_classes=4,
                n_informative=3,
            )
            y = np.vstack([y, y, y]).T
            return x, y  # type: ignore
        case "continuous":
            return make_regression(random_state=42, n_samples=20, n_targets=1)  # type: ignore
        case "continuous-multioutput":
            return make_regression(random_state=42, n_samples=20, n_targets=2)  # type: ignore

    raise ValueError(f"Unknown task type {task_type}")


def _sample_y(task_type: TaskTypeName) -> np.ndarray:
    return data_for_task_type(task_type)[1]


@parametrize(
    "real, task_hint, expected",
    [
        ("binary", None, "binary"),
        ("binary", "classification", "binary"),
        ("binary", "regression", "continuous"),
        #
        ("multiclass", None, "multiclass"),
        ("multiclass", "classification", "multiclass"),
        ("multiclass", "regression", "continuous"),
        #
        ("multilabel-indicator", None, "multilabel-indicator"),
        ("multilabel-indicator", "classification", "multilabel-indicator"),
        ("multilabel-indicator", "regression", "continuous-multioutput"),
        #
        ("multiclass-multioutput", None, "multiclass-multioutput"),
        ("multiclass-multioutput", "classification", "multiclass-multioutput"),
        ("multiclass-multioutput", "regression", "continuous-multioutput"),
        #
        ("continuous", None, "continuous"),
        ("continuous", "classification", "multiclass"),
        ("continuous", "regression", "continuous"),
        #
        ("continuous-multioutput", None, "continuous-multioutput"),
        ("continuous-multioutput", "classification", "multiclass-multioutput"),
        ("continuous-multioutput", "regression", "continuous-multioutput"),
    ],
)
def test_identify_task_type(
    real: TaskTypeName,
    task_hint: Literal["classification", "regression"] | None,
    expected: TaskTypeName,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TaskTypeWarning)

        if real == "continuous-multioutput" and task_hint == "classification":
            # Special case since we have to check when the y with multiple values.
            A, B, C = 0.1, 2.1, 3.1

            # <=2 unique values per column
            y = np.array(
                [
                    [A, B],
                    [A, B],
                    [A, C],
                ],
            )

            identified = identify_task_type(y, task_hint=task_hint)
            assert identified == "multilabel-indicator"

            # >2 unique values per column
            y = np.array(
                [
                    [A, A],
                    [B, B],
                    [C, C],
                ],
            )
            identified = identify_task_type(y, task_hint=task_hint)
            assert identified == "multiclass-multioutput"

        elif real == "continuous" and task_hint == "classification":
            # Special case since we have to check when the y with multiple values.
            A, B, C = 0.1, 2.1, 3.1

            y = np.array([A, A, B, B])
            identified = identify_task_type(y, task_hint=task_hint)
            assert identified == "binary"

            y = np.array([A, B, C])
            identified = identify_task_type(y, task_hint=task_hint)
            assert identified == "multiclass"
        else:
            y = _sample_y(expected)
            identified = identify_task_type(y, task_hint=task_hint)

            assert identified == expected


@parametrize(
    "task_type, expected",
    [
        # Holdout
        ("binary", StratifiedShuffleSplit),
        ("multiclass", StratifiedShuffleSplit),
        ("multilabel-indicator", ShuffleSplit),
        ("multiclass-multioutput", ShuffleSplit),
        ("continuous", ShuffleSplit),
        ("continuous-multioutput", ShuffleSplit),
    ],
)
def test_default_holdout(task_type: TaskTypeName, expected: type) -> None:
    sampler = _default_holdout(task_type, holdout_size=0.387, random_state=42)
    assert isinstance(sampler, expected)
    assert sampler.n_splits == 1  # type: ignore
    assert sampler.random_state == 42  # type: ignore
    assert sampler.test_size == 0.387  # type: ignore


@parametrize(
    "task_type, expected",
    [
        # CV - Notable, only binary and multiclass can be stratified
        ("binary", StratifiedKFold),
        ("multiclass", StratifiedKFold),
        ("multilabel-indicator", KFold),
        ("multiclass-multioutput", KFold),
        ("continuous", KFold),
        ("continuous-multioutput", KFold),
    ],
)
def test_default_resampling(task_type: TaskTypeName, expected: type) -> None:
    sampler = _default_cv_resampler(task_type, n_splits=2, random_state=42)
    assert isinstance(sampler, expected)
    assert sampler.n_splits == 2  # type: ignore
    assert sampler.random_state == 42  # type: ignore


@dataclass
class _EvalKwargs:
    trial: Trial
    pipeline: Component
    additional_scorers: Mapping[str, _Scorer] | None
    params: Mapping[str, Any] | None
    task_type: TaskTypeName
    working_dir: Path
    X: pd.DataFrame | np.ndarray
    y: pd.Series | np.ndarray | pd.DataFrame


@case
@parametrize(
    "metric",
    [
        # Single ob
        Metric("accuracy", minimize=False, bounds=(0, 1)),
        # Mutli obj
        [
            Metric("custom", minimize=False, bounds=(0, 1), fn=get_scorer("accuracy")),
            Metric("roc_auc_ovr", minimize=False, bounds=(0, 1)),
        ],
    ],
)
@parametrize(
    "additional_scorers",
    [
        None,
        {"acc": get_scorer("accuracy"), "roc": get_scorer("roc_auc_ovr")},
    ],
)
@parametrize(
    "task_type",
    ["binary", "multiclass", "multilabel-indicator"],
)
def case_classification(
    tmp_path: Path,
    metric: Metric | list[Metric],
    additional_scorers: Mapping[str, _Scorer] | None,
    task_type: TaskTypeName,
) -> _EvalKwargs:
    x, y = data_for_task_type(task_type)
    return _EvalKwargs(
        trial=Trial.create(
            name="test",
            config={},
            seed=42,
            bucket=tmp_path / "trial",
            metrics=metric,
        ),
        task_type=task_type,
        pipeline=Component(DecisionTreeClassifier, config={"max_depth": 1}),
        additional_scorers=additional_scorers,
        params=None,
        working_dir=tmp_path / "data",
        X=x,
        y=y,
    )


@case
@parametrize(
    "metric",
    [
        # Single ob
        Metric("neg_mean_absolute_error", minimize=True, bounds=(-np.inf, 0)),
        # Mutli obj
        [
            Metric("custom", minimize=False, bounds=(-np.inf, 1), fn=get_scorer("r2")),
            Metric("neg_mean_squared_error", minimize=False, bounds=(-np.inf, 0)),
        ],
    ],
)
@parametrize(
    "additional_scorers",
    [
        None,
        {
            "rmse": get_scorer("neg_root_mean_squared_error"),
            "err": get_scorer("neg_mean_absolute_error"),
        },
    ],
)
@parametrize("task_type", ["continuous", "continuous-multioutput"])
def case_regression(
    tmp_path: Path,
    metric: Metric | list[Metric],
    additional_scorers: Mapping[str, _Scorer] | None,
    task_type: TaskTypeName,
) -> _EvalKwargs:
    x, y = data_for_task_type(task_type)

    return _EvalKwargs(
        trial=Trial.create(
            name="test",
            config={},
            seed=42,
            bucket=tmp_path / "trial",
            metrics=metric,
        ),
        pipeline=Component(DecisionTreeRegressor, config={"max_depth": 1}),
        additional_scorers=additional_scorers,
        task_type=task_type,
        params=None,
        working_dir=tmp_path / "data",
        X=x,
        y=y,
    )


@parametrize("as_pd", [True, False])
@parametrize("store_models", [True, False])
@parametrize("train_score", [True, False])
@parametrize_with_cases("item", cases=".", prefix="case_")
@parametrize("cv_value, strategy", [(2, "cv"), (0.3, "holdout")])
def test_evaluator(
    as_pd: bool,
    store_models: bool,
    train_score: bool,
    item: _EvalKwargs,
    cv_value: int | float,
    strategy: str,
) -> None:
    x = pd.DataFrame(item.X) if as_pd else item.X
    y = (
        item.y
        if not as_pd
        else (pd.DataFrame(item.y) if np.ndim(item.y) > 1 else pd.Series(item.y))
    )
    trial = item.trial
    if strategy == "cv":
        cv_kwargs = {"n_splits": cv_value, "strategy": "cv"}
    else:
        cv_kwargs = {"holdout_size": cv_value, "strategy": "holdout"}

    evaluator = CVEvaluation(
        X=x,
        y=y,
        working_dir=item.working_dir,
        train_score=train_score,
        store_models=store_models,
        params=item.params,
        additional_scorers=item.additional_scorers,
        task_hint=item.task_type,
        random_state=42,
        on_error="raise",
        **cv_kwargs,  # type: ignore
    )
    n_splits = evaluator.splitter.get_n_splits(x, y)
    assert n_splits is not None

    report = evaluator.fn(trial, item.pipeline)

    # ------- Property testing

    # Model should be stored
    if store_models:
        for i in range(n_splits):
            assert f"model_{i}.pkl" in report.storage

    # All metrics should be recorded and valid
    for metric_name, metric in trial.metrics.items():
        assert metric_name in report.values
        value = report.values[metric_name]
        # ... in correct bounds
        if metric.bounds is not None:
            assert metric.bounds[0] <= value <= metric.bounds[1]

    # Summary should contain all optimization metrics
    expected_summary_scorers = [
        *trial.metrics.keys(),
        *(item.additional_scorers.keys() if item.additional_scorers else []),
    ]
    for metric_name in expected_summary_scorers:
        for i in range(n_splits):
            assert f"fold_{i}:{metric_name}" in report.summary
        assert f"mean_{metric_name}" in report.summary
        assert f"std_{metric_name}" in report.summary

    if train_score:
        for metric_name in expected_summary_scorers:
            for i in range(n_splits):
                assert f"fold_{i}:train_{metric_name}" in report.summary
            assert f"train_mean_{metric_name}" in report.summary
            assert f"train_std_{metric_name}" in report.summary

    # All folds are profiled
    assert "cv" in report.profiles
    for i in range(n_splits):
        assert f"cv:fold_{i}" in report.profiles


@parametrize(
    "task_type",
    [
        "binary",
        "multiclass",
        "multilabel-indicator",
        "multiclass-multioutput",
        "continuous",
        "continuous-multioutput",
    ],
)
@parametrize("cv_value, strategy", [(2, "cv"), (0.3, "holdout")])
def test_consistent_results_across_seeds(
    tmp_path: Path,
    cv_value: int | float,
    strategy: Literal["cv", "holdout"],
    task_type: TaskTypeName,
) -> None:
    x, y = data_for_task_type(task_type)
    match task_type:
        case "binary" | "multiclass" | "multilabel-indicator":
            pipeline = Component(
                DecisionTreeClassifier,
                config={"max_depth": 1, "random_state": request("random_state")},
            )
            metric = Metric("accuracy", minimize=False, bounds=(0, 1))
        case "continuous" | "continuous-multioutput":
            pipeline = Component(
                DecisionTreeRegressor,
                config={"max_depth": 1, "random_state": request("random_state")},
            )
            metric = Metric("r2", minimize=True, bounds=(-np.inf, 1))
        case "multiclass-multioutput":
            pipeline = Component(
                DecisionTreeClassifier,
                config={"max_depth": 1, "random_state": request("random_state")},
            )
            # Sklearn doesn't have any multiclass-multioutput metrics
            metric = Metric(
                "custom",
                minimize=False,
                bounds=(0, 1),
                fn=lambda y_true, y_pred: (y_pred == y_true).mean().mean(),
            )

    if strategy == "cv":
        cv_kwargs = {"n_splits": cv_value, "strategy": "cv"}
    else:
        cv_kwargs = {"holdout_size": cv_value, "strategy": "holdout"}

    evaluator_1 = CVEvaluation(
        X=x,
        y=y,
        working_dir=tmp_path,
        random_state=42,
        train_score=True,
        store_models=False,
        task_hint=task_type,
        params=None,
        on_error="raise",
        **cv_kwargs,  # type: ignore
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitMetricConversionWarning)

        report_1 = evaluator_1.fn(
            Trial.create(
                name="trial-name",
                config={},
                seed=50,
                bucket=tmp_path / "trial-name",
                metrics=metric,
            ),
            pipeline,
        )

    # Make sure to clean up the bucket for the second
    # trial as it will have the same name
    report_1.bucket.rmdir()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitMetricConversionWarning)

        report_2 = evaluator_1.fn(
            Trial.create(
                name="trial-name",
                config={},
                seed=50,
                bucket=tmp_path / "trial-name",  # We give a different dir
                metrics=metric,
            ),
            pipeline,
        )

    # We ignore profiles because they will be different timings
    # We report.reported_at as they will naturally be different
    df_1 = report_1.df(profiles=False).drop(columns=["reported_at"])
    df_2 = report_2.df(profiles=False).drop(columns=["reported_at"])
    pd.testing.assert_frame_equal(df_1, df_2)
