from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import sklearn.pipeline
from pytest_cases import case, parametrize, parametrize_with_cases
from sklearn import config_context as sklearn_config_context
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer, make_scorer
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amltk.exceptions import TaskTypeWarning, TrialError
from amltk.optimization.trial import Metric, Trial
from amltk.pipeline import Component, request
from amltk.pipeline.builders.sklearn import build as sklearn_pipeline_builder
from amltk.sklearn.evaluation import (
    CVEvaluation,
    ImplicitMetricConversionWarning,
    TaskTypeName,
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
        ("binary", "auto", "binary"),
        ("binary", "classification", "binary"),
        ("binary", "regression", "continuous"),
        #
        ("multiclass", "auto", "multiclass"),
        ("multiclass", "classification", "multiclass"),
        ("multiclass", "regression", "continuous"),
        #
        ("multilabel-indicator", "auto", "multilabel-indicator"),
        ("multilabel-indicator", "classification", "multilabel-indicator"),
        ("multilabel-indicator", "regression", "continuous-multioutput"),
        #
        ("multiclass-multioutput", "auto", "multiclass-multioutput"),
        ("multiclass-multioutput", "classification", "multiclass-multioutput"),
        ("multiclass-multioutput", "regression", "continuous-multioutput"),
        #
        ("continuous", "auto", "continuous"),
        ("continuous", "classification", "multiclass"),
        ("continuous", "regression", "continuous"),
        #
        ("continuous-multioutput", "auto", "continuous-multioutput"),
        ("continuous-multioutput", "classification", "multiclass-multioutput"),
        ("continuous-multioutput", "regression", "continuous-multioutput"),
    ],
)
def test_identify_task_type(
    real: TaskTypeName,
    task_hint: Literal["classification", "regression", "auto"],
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
@parametrize("cv_value, splitter", [(2, "cv"), (0.3, "holdout")])
def test_evaluator(
    as_pd: bool,
    store_models: bool,
    train_score: bool,
    item: _EvalKwargs,
    cv_value: int | float,
    splitter: str,
) -> None:
    x = pd.DataFrame(item.X) if as_pd else item.X
    y = (
        item.y
        if not as_pd
        else (pd.DataFrame(item.y) if np.ndim(item.y) > 1 else pd.Series(item.y))
    )
    trial = item.trial
    if splitter == "cv":
        cv_kwargs = {"n_splits": cv_value, "splitter": "cv"}
    else:
        cv_kwargs = {"holdout_size": cv_value, "splitter": "holdout"}

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
@parametrize("cv_value, splitter", [(2, "cv"), (0.3, "holdout")])
def test_consistent_results_across_seeds(
    tmp_path: Path,
    cv_value: int | float,
    splitter: Literal["cv", "holdout"],
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

    if splitter == "cv":
        cv_kwargs = {"n_splits": cv_value, "splitter": "cv"}
    else:
        cv_kwargs = {"holdout_size": cv_value, "splitter": "holdout"}

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
    # We ignore trial.created_at and report.reported_at as they will naturally
    # be different
    df_1 = report_1.df(profiles=False).drop(columns=["reported_at", "created_at"])
    df_2 = report_2.df(profiles=False).drop(columns=["reported_at", "created_at"])
    pd.testing.assert_frame_equal(df_1, df_2)


def test_scoring_params_get_forwarded(tmp_path: Path) -> None:
    with sklearn_config_context(enable_metadata_routing=True):
        pipeline = Component(DecisionTreeClassifier, config={"max_depth": 1})
        x, y = data_for_task_type("binary")

        # This custom metrics requires a custom parameter
        def custom_metric(
            y_true: np.ndarray,  # noqa: ARG001
            y_pred: np.ndarray,  # noqa: ARG001
            *,
            scorer_param_required: float,
        ):
            return scorer_param_required

        custom_scorer = (
            make_scorer(custom_metric, response_method="predict")
            # Here we specify that it needs this parameter routed to it
            .set_score_request(scorer_param_required=True)
        )

        value = 0.123
        evaluator = CVEvaluation(
            x,
            y,
            params={"scorer_param_required": value},  # Pass it in here
            working_dir=tmp_path,
            on_error="raise",
        )
        trial = Trial.create(
            name="test",
            bucket=tmp_path / "trial",
            metrics=Metric(name="custom_metric", fn=custom_scorer),
        )
        report = evaluator.fn(trial, pipeline)

        assert report.values["custom_metric"] == value


def test_splitter_params_get_forwarded(tmp_path: Path) -> None:
    with sklearn_config_context(enable_metadata_routing=True):
        # A DecisionTreeClassifier by default allows for sample_weight as a parameter
        # request
        pipeline = Component(DecisionTreeClassifier, config={"max_depth": 1})
        x, y = data_for_task_type("binary")

        # Make a group which is half 0 and half 1
        _half = len(x) // 2
        fake_groups = np.asarray([0] * _half + [1] * (len(x) - _half))

        trial = Trial.create(name="test", bucket=tmp_path / "trial")

        # First make sure it errors if groups is not provided to the splitter
        evaluator = CVEvaluation(
            x,
            y,
            # params={"groups": fake_groups},  # noqa: ERA001
            splitter=GroupKFold(n_splits=2),
            working_dir=tmp_path,
            on_error="raise",
        )
        with pytest.raises(
            TrialError,
            match=r"The 'groups' parameter should not be None.",
        ):
            evaluator.fn(trial, pipeline)

        # Now make sure it works
        evaluator = CVEvaluation(
            x,
            y,
            splitter=GroupKFold(n_splits=2),  # We specify a group splitter
            params={"groups": fake_groups},  # Pass it in here
            working_dir=tmp_path,
            on_error="raise",
        )
        evaluator.fn(trial, pipeline)


def test_estimator_params_get_forward(tmp_path: Path) -> None:
    with sklearn_config_context(enable_metadata_routing=True):
        # NOTE: There is no way to explcitly check that metadata was indeed
        # routed to the estimator, e.g. through an error. Please see this
        # issue
        # https://github.com/scikit-learn/scikit-learn/issues/23920

        # We'll test this using the DummyClassifier with a Prior config.
        # Thankfully this is deterministic so it's attributes_ should
        # only get modified based on it's input.
        # One attribute_ that gets modified depending on sample_weight
        # is estimator.class_prior_ which we can check pretty easily.
        x, y = data_for_task_type("binary")
        sample_weight = np.random.rand(len(x))  # noqa: NPY002

        def create_dummy_classifier_with_sample_weight_request(
            *args: Any,
            **kwargs: Any,
        ) -> DummyClassifier:
            est = DummyClassifier(*args, **kwargs)
            # https://scikit-learn.org/stable/metadata_routing.html#api-interface
            est.set_fit_request(sample_weight=True)
            return est

        pipeline = Component(
            create_dummy_classifier_with_sample_weight_request,
            config={"strategy": "prior"},
        )

        # First we use an evaluator without sample_weight
        trial = Trial.create(name="test", bucket=tmp_path / "trial_1")
        evaluator = CVEvaluation(
            x,
            y,
            holdout_size=0.3,
            working_dir=tmp_path,
            store_models=True,
            # params={"sample_weight": sample_weight},  # noqa: ERA001
            on_error="raise",
        )
        report = evaluator.fn(trial, pipeline)

        # load pipeline, get 0th model, get it's class_prior_
        class_weights_1 = report.retrieve("model_0.pkl")[0].class_prior_

        # To make sure that our tests are correct, we repeat this without
        # sample weights, should remain the same
        trial = Trial.create(name="test", bucket=tmp_path / "trial_2")
        report = evaluator.fn(trial, pipeline)
        class_weights_2 = report.retrieve("model_0.pkl")[0].class_prior_

        np.testing.assert_array_equal(class_weights_1, class_weights_2)

        # Now with the sample weights, the class_prior_ should be different
        trial = Trial.create(name="test", bucket=tmp_path / "trial_3")
        evaluator = CVEvaluation(
            x,
            y,
            holdout_size=0.3,
            working_dir=tmp_path,
            store_models=True,
            params={"sample_weight": sample_weight},  # Passed in this time
            on_error="raise",
        )
        report = evaluator.fn(trial, pipeline)
        class_weights_3 = report.retrieve("model_0.pkl")[0].class_prior_

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(class_weights_1, class_weights_3)


def test_evaluator_with_clustering(tmp_path: Path) -> None:
    x, y = sklearn.datasets.make_blobs(
        n_samples=20,
        centers=2,
        n_features=2,
        random_state=42,
    )
    pipeline = Component(KMeans, config={"n_clusters": 2, "random_state": 42})

    metrics = Metric("adjusted_rand_score", minimize=False, bounds=(-0.5, 1))
    trial = Trial.create(name="test", bucket=tmp_path / "trial", metrics=metrics)

    evaluator = CVEvaluation(
        x,  # type: ignore
        y,  # type: ignore
        working_dir=tmp_path,
        on_error="raise",
        random_state=42,
    )
    report = evaluator.fn(trial, pipeline)

    # We are not really trying to detect the score of the algorithm, just to ensure
    # that it did indeed train with the data and does not error.
    # If it seems to get a slightly less score than 1.0 then that's okay,
    # just change the value. Should not change due to the seeding but
    # make sklearn changes something
    assert "adjusted_rand_score" in report.values
    assert report.values["adjusted_rand_score"] == pytest.approx(1.0)


@pytest.mark.xfail(
    reason=(
        "This was a bug introduced by sklearn. If this fails, "
        " it can be safely ignored for now. If it starts to work"
        " on the action runners, please remove this xfail."
        " See https://github.com/scikit-learn/scikit-learn/pull/28371"
    ),
)
def test_custom_configure_gets_forwarded(tmp_path: Path) -> None:
    with sklearn_config_context(enable_metadata_routing=True):
        # Pipeline requests a max_depth, defaulting to 1
        pipeline = Component(
            DecisionTreeClassifier,
            config={
                "max_depth": request("max_depth", default=1),
            },
        )

        # We pass in explicitly to configure with 2
        # This can be useful for estimators that require explicit information
        # about the dataset
        configure_params = {"max_depth": 2}

        x, y = data_for_task_type("binary")
        evaluator = CVEvaluation(
            x,
            y,
            params={"configure": configure_params},
            working_dir=tmp_path,
            splitter="holdout",
            holdout_size=0.3,
            store_models=True,
            on_error="raise",
        )
        trial = Trial.create(
            name="test",
            bucket=tmp_path / "trial",
            metrics=Metric("accuracy"),
        )
        report = evaluator.fn(trial, pipeline)
        model = report.retrieve("model_0.pkl")[0]
        assert model.max_depth == 2


# Used in the test below
class _MyPipeline(sklearn.pipeline.Pipeline):
    # Have to explcitiyl list out all parameters by sklearn API
    def __init__(
        self,
        steps: Any,
        *,
        memory: None = None,
        verbose: bool = False,
        bamboozled: str = "no",
    ):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.bamboozled = bamboozled


# Used in test below, builds one of the
# _MyPipeline with a custom parameter that
# will also get passed in
def _my_custom_builder(
    *args: Any,
    bamboozled: str = "no",
    **kwargs: Any,
) -> _MyPipeline:
    return sklearn_pipeline_builder(
        *args,
        pipeline_type=_MyPipeline,
        bamboozled=bamboozled,
        **kwargs,
    )


@pytest.mark.xfail(
    reason=(
        "This was a bug introduced by sklearn. If this fails, "
        " it can be safely ignored for now. If it starts to work"
        " on the action runners, please remove this xfail."
        " See https://github.com/scikit-learn/scikit-learn/pull/28371"
    ),
)
def test_custom_builder_can_be_forwarded(tmp_path: Path) -> None:
    with sklearn_config_context(enable_metadata_routing=True):
        pipeline = Component(DecisionTreeClassifier, config={"max_depth": 1})

        x, y = data_for_task_type("binary")
        evaluator = CVEvaluation(
            x,
            y,
            params={"build": {"builder": _my_custom_builder, "bamboozled": "yes"}},
            working_dir=tmp_path,
            store_models=True,
            on_error="raise",
        )
        trial = Trial.create(
            name="test",
            bucket=tmp_path / "trial",
            metrics=Metric("accuracy"),
        )

        report = evaluator.fn(trial, pipeline)
        model = report.retrieve("model_0.pkl")
        assert isinstance(model, _MyPipeline)
        assert hasattr(model, "bamboozled")
        assert model.bamboozled == "yes"
