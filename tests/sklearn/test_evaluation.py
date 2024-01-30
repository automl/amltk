from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pytest_cases import case, parametrize, parametrize_with_cases
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amltk.optimization.trial import Metric, Trial
from amltk.pipeline import Component
from amltk.sklearn.evaluation import (
    CVEvaluation,
    TaskTypeName,
    _default_resampler,
    identify_task_type,
)


def data_for_task_type(task_type: TaskTypeName) -> tuple[np.ndarray, np.ndarray]:
    match task_type:
        case "binary":
            return make_classification(random_state=42, n_classes=2, n_informative=3)  # type: ignore
        case "multiclass":
            return make_classification(random_state=42, n_classes=4, n_informative=3)  # type: ignore
        case "multilabel-indicator":
            x, y = make_classification(random_state=42, n_classes=2, n_informative=3)
            y = np.vstack([y, y]).T
            return x, y  # type: ignore
        case "multiclass-multioutput":
            x, y = make_classification(random_state=42, n_classes=4, n_informative=3)
            y = np.vstack([y, y, y]).T
            return x, y  # type: ignore
        case "continuous":
            return make_regression(random_state=42, n_targets=1)  # type: ignore
        case "continuous-multioutput":
            return make_regression(random_state=42, n_targets=2)  # type: ignore


def _sample_y(task_type: TaskTypeName) -> np.ndarray:
    return data_for_task_type(task_type)[1]


@parametrize(
    "y, is_classification, expected",
    [
        # with is_classification = None
        # --- classification
        (_sample_y("binary"), None, "binary"),
        (_sample_y("multiclass"), None, "multiclass"),
        (_sample_y("multilabel-indicator"), None, "multilabel-indicator"),
        (_sample_y("multiclass-multioutput"), None, "multiclass-multioutput"),
        # --- regression
        (_sample_y("continuous"), None, "continuous"),
        (_sample_y("continuous-multioutput"), None, "continuous-multioutput"),
        # with is_classification = True
        # --- classification
        (_sample_y("binary"), True, "binary"),
        (_sample_y("multiclass"), True, "multiclass"),
        (_sample_y("multilabel-indicator"), True, "multilabel-indicator"),
        (_sample_y("multiclass-multioutput"), True, "multiclass-multioutput"),
        # --- regression
        (_sample_y("multiclass"), True, "multiclass"),
        (_sample_y("multiclass-multioutput"), True, "multiclass-multioutput"),
        # with is_classification = False
        # --- classification
        (_sample_y("continuous"), False, "continuous"),
        (_sample_y("continuous"), False, "continuous"),
        (_sample_y("continuous-multioutput"), False, "continuous-multioutput"),
        (_sample_y("continuous-multioutput"), False, "continuous-multioutput"),
        # --- regression
        (_sample_y("continuous"), False, "continuous"),
        (_sample_y("continuous-multioutput"), False, "continuous-multioutput"),
    ],
)
def test_identify_task_type(
    y: np.ndarray,
    is_classification: bool,
    expected: str,
) -> None:
    identified = identify_task_type(y, is_classification=is_classification)
    assert identified == expected

    pd_y = pd.Series(y) if np.ndim(y) == 1 else pd.DataFrame(y)
    identified = identify_task_type(pd_y, is_classification=is_classification)
    assert identified == expected


@parametrize(
    "task_type, n_splits, train_size, expected",
    [
        # CV - Notable, only binary and multiclass can be stratified
        ("binary", 3, 0.0, StratifiedKFold),
        ("multiclass", 3, 0.0, StratifiedKFold),
        ("multilabel-indicator", 3, 0.0, KFold),
        ("multiclass-multioutput", 3, 0.0, KFold),
        ("continuous", 3, 0.0, KFold),
        ("continuous-multioutput", 3, 0.0, KFold),
        # Holdout
        ("binary", 1, 0.3, StratifiedShuffleSplit),  # With only one fold and test size
        ("multiclass", 1, 0.3, StratifiedShuffleSplit),
        ("multilabel-indicator", 1, 0.3, ShuffleSplit),
        ("multiclass-multioutput", 1, 0.3, ShuffleSplit),
        ("continuous", 1, 0.3, ShuffleSplit),
        ("continuous-multioutput", 1, 0.3, ShuffleSplit),
    ],
)
def test_default_resampling(
    task_type: TaskTypeName,
    n_splits: int,
    train_size: float,
    expected: type,
) -> None:
    sampler = _default_resampler(
        task_type,
        n_splits=n_splits,
        random_state=42,
        train_size=train_size,
    )
    assert isinstance(sampler, expected)
    assert sampler.n_splits == n_splits  # type: ignore
    assert sampler.random_state == 42  # type: ignore

    if train_size != 0.0:
        assert sampler.train_size == train_size  # type: ignore


@dataclass
class _EvalutionCase:
    trial: Trial
    pipeline: Component
    cv: int | float | BaseShuffleSplit | BaseCrossValidator
    additional_scorers: Mapping[str, _Scorer] | None
    params: Mapping[str, Any] | None
    datadir: Path
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
@parametrize("cv", [3, 0.3])
def case_classification(
    tmp_path: Path,
    metric: Metric | list[Metric],
    additional_scorers: Mapping[str, _Scorer] | None,
    task_type: TaskTypeName,
    cv: int | float | BaseShuffleSplit | BaseCrossValidator,
) -> _EvalutionCase:
    x, y = data_for_task_type(task_type)
    return _EvalutionCase(
        trial=Trial.create(
            name="test",
            config={},
            seed=42,
            bucket=tmp_path / "trial",
            metrics=metric,
        ),
        pipeline=Component(DecisionTreeClassifier, config={"max_depth": 1}),
        cv=cv,
        additional_scorers=additional_scorers,
        params=None,
        datadir=tmp_path / "data",
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
@parametrize("cv", [3, 0.3])
def case_regression(
    tmp_path: Path,
    metric: Metric | list[Metric],
    additional_scorers: Mapping[str, _Scorer] | None,
    task_type: TaskTypeName,
    cv: int | float | BaseShuffleSplit | BaseCrossValidator,
) -> _EvalutionCase:
    x, y = data_for_task_type(task_type)
    return _EvalutionCase(
        trial=Trial.create(
            name="test",
            config={},
            seed=42,
            bucket=tmp_path / "trial",
            metrics=metric,
        ),
        pipeline=Component(DecisionTreeRegressor, config={"max_depth": 1}),
        cv=cv,
        additional_scorers=additional_scorers,
        params=None,
        datadir=tmp_path / "data",
        X=x,
        y=y,
    )


@parametrize("as_pd", [True, False])
@parametrize("store_models", [True, False])
@parametrize("train_score", [True, False])
@parametrize_with_cases("item", cases=".", prefix="case_")
def test_evaluator(
    as_pd: bool,
    store_models: bool,
    train_score: bool,
    item: _EvalutionCase,
) -> None:
    x = pd.DataFrame(item.X) if as_pd else item.X
    y = (
        item.y
        if not as_pd
        else (pd.DataFrame(item.y) if np.ndim(item.y) > 1 else pd.Series(item.y))
    )
    trial = item.trial

    evaluator = CVEvaluation(
        X=x,
        y=y,
        cv=item.cv,
        datadir=item.datadir,
        train_score=train_score,
        store_models=store_models,
        params=item.params,
        additional_scorers=item.additional_scorers,
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
