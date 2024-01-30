"""This module contains the cross-validation evaluation protocol.

This protocol will create a cross-validation task to be used in parallel and
optimization. It represents a typical cross-validation evaluation for sklearn,
handling some of the minor nuances of sklearn and it's interaction with
optimization and parallelization.

Please see [`CVEvaluation`][amltk.sklearn.evaluation.CVEvaluation] for more
information on usage.
"""
from __future__ import annotations

import logging
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, TypeVar
from typing_extensions import override

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _MultimetricScorer, _Scorer
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.model_selection._validation import _score
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_method_params

import amltk.randomness
from amltk._functional import subclass_map
from amltk.exceptions import (
    AutomaticTaskTypeInferredWarning,
    ImplicitMetricConversionWarning,
    MismatchedTaskTypeWarning,
)
from amltk.optimization.evaluation import EvaluationProtocol
from amltk.profiling.profiler import Profiler
from amltk.store import Stored
from amltk.store.paths.path_bucket import PathBucket

if TYPE_CHECKING:
    from sklearn.model_selection import (
        BaseCrossValidator,
        BaseShuffleSplit,
    )
    from sklearn.utils import Bunch

    from amltk.optimization import Trial
    from amltk.pipeline import Node
    from amltk.randomness import Seed
    from amltk.scheduling import Plugin, Scheduler, Task


logger = logging.getLogger(__name__)

BaseEstimatorT = TypeVar("BaseEstimatorT", bound=BaseEstimator)
TaskTypeName: TypeAlias = Literal[
    "binary",
    "multiclass",
    "multilabel-indicator",
    "multiclass-multioutput",
    "continuous",
    "continuous-multioutput",
]
_valid_task_types: tuple[TaskTypeName, ...] = (
    "binary",
    "multiclass",
    "multilabel-indicator",
    "multiclass-multioutput",
    "continuous",
    "continuous-multioutput",
)


def _route_params(
    splitter: BaseShuffleSplit | BaseCrossValidator,
    estimator: BaseEstimator,
    _scorer: _Scorer | _MultimetricScorer,
    **params: Any,
) -> Bunch:
    # NOTE: This is basically copied out of sklearns 1.4 cross_validate

    # For estimators, a MetadataRouter is created in get_metadata_routing
    # methods. For these router methods, we create the router to use
    # `process_routing` on it.
    router = (
        MetadataRouter(owner="cross_validate")
        .add(
            splitter=splitter,
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )
        .add(
            estimator=estimator,
            # TODO(SLEP6): also pass metadata to the predict method for
            # scoring?
            # ^ Taken from cross_validate source code in sklearn
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        .add(
            scorer=_scorer,
            method_mapping=MethodMapping().add(caller="fit", callee="score"),
        )
    )
    try:
        return process_routing(router, "fit", **params)  # type: ignore
    except UnsetMetadataPassedError as e:
        # The default exception would mention `fit` since in the above
        # `process_routing` code, we pass `fit` as the caller. However,
        # the user is not calling `fit` directly, so we change the message
        # to make it more suitable for this case.
        raise UnsetMetadataPassedError(
            message=(
                f"{sorted(e.unrequested_params.keys())} are passed to cross"
                " validation but are not explicitly requested or unrequested. See"
                " the Metadata Routing User guide"
                " <https://scikit-learn.org/stable/metadata_routing.html> for more"
                " information."
            ),
            unrequested_params=e.unrequested_params,
            routed_params=e.routed_params,
        ) from e


def _default_holdout(
    task_type: TaskTypeName,
    holdout_size: float,
    *,
    random_state: Seed | None = None,
) -> ShuffleSplit | StratifiedShuffleSplit:
    if not (0 < holdout_size < 1):
        raise ValueError(f"`{holdout_size=}` must be in (0, 1)")

    rs = amltk.randomness.as_int(random_state)
    match task_type:
        case "binary" | "multiclass":
            return StratifiedShuffleSplit(1, random_state=rs, test_size=holdout_size)
        case "multilabel-indicator" | "multiclass-multioutput":
            return ShuffleSplit(1, random_state=rs, test_size=holdout_size)
        case "continuous" | "continuous-multioutput":
            return ShuffleSplit(1, random_state=rs, test_size=holdout_size)
        case _:
            raise ValueError(f"Don't know how to handle {task_type=}")


def _default_cv_resampler(
    task_type: TaskTypeName,
    n_splits: int,
    *,
    random_state: Seed | None = None,
) -> StratifiedKFold | KFold:
    if n_splits < 1:
        raise ValueError(f"Must have at least one split but got {n_splits=}")

    rs = amltk.randomness.as_int(random_state)

    match task_type:
        case "binary" | "multiclass":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        case "multilabel-indicator" | "multiclass-multioutput":
            # NOTE: They don't natively support multilabel-indicator for stratified
            return KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        case "continuous" | "continuous-multioutput":
            return KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        case _:
            raise ValueError(f"Don't know how to handle {task_type=} with {n_splits=}")


def identify_task_type(  # noqa: PLR0911
    y: np.ndarray | pd.Series | pd.DataFrame,
    *,
    task_hint: Literal["classification", "regression", "auto"] = "auto",
) -> TaskTypeName:
    """Identify the task type from the target data."""
    inferred_type: TaskTypeName = type_of_target(y)
    if task_hint == "auto":
        warnings.warn(
            f"`{task_hint=}` was not provided. The task type was inferred from"
            f" the target data to be '{inferred_type}'."
            " To silence this warning, please provide `task_hint`.",
            AutomaticTaskTypeInferredWarning,
            stacklevel=2,
        )
        return inferred_type

    match task_hint, inferred_type:
        # First two cases are everything is fine
        case (
            "classification",
            "binary"
            | "multiclass"
            | "multilabel-indicator"
            | "multiclass-multioutput",
        ):
            return inferred_type
        case ("regression", "continuous" | "continuous-multioutput"):
            return inferred_type
        # Hinted to be regression but we got a single column classification task
        case ("regression", "binary" | "multiclass"):
            warnings.warn(
                f"`{task_hint=}` but `{inferred_type=}`."
                " Set to `continuous` as there is only one target column.",
                MismatchedTaskTypeWarning,
                stacklevel=2,
            )
            return "continuous"
        # Hinted to be regression but we got multi-column classification task
        case ("regression", "multilabel-indicator" | "multiclass-multioutput"):
            warnings.warn(
                f"`{task_hint=}` but `{inferred_type=}`."
                " Set to `continuous-multiouput` as there are more than 1 target"
                " columns.",
                MismatchedTaskTypeWarning,
                stacklevel=2,
            )
            return "continuous"
        # Hinted to be classification but we got a single column regression task
        case ("classification", "continuous"):
            match len(np.unique(y)):
                case 1:
                    raise ValueError(
                        "The target data has only one unique value. This is"
                        f" not a valid classification task.\n{y=}",
                    )
                case 2:
                    warnings.warn(
                        f"`{task_hint=}` but `{inferred_type=}`."
                        " Set to `binary` as only 2 unique values."
                        " To silence this, provide a specific task type to"
                        f"`task_hint=` from {_valid_task_types}.",
                        MismatchedTaskTypeWarning,
                        stacklevel=2,
                    )
                    return "binary"
                case _:
                    warnings.warn(
                        f"`{task_hint=}` but `{inferred_type=}`."
                        " Set to `multiclass` as >2 unique values."
                        " To silence this, provide a specific task type to"
                        f"`task_hint=` from {_valid_task_types}.",
                        MismatchedTaskTypeWarning,
                        stacklevel=2,
                    )
                    return "multiclass"
        # Hinted to be classification but we got multi-column regression task
        case ("classification", "continuous-multioutput"):
            # NOTE: this is a matrix wide .unique, I'm not sure how things
            # work with multiclass-multioutput and whether it should be
            # done by 2 unique per column
            uniques_per_col = [np.unique(col) for col in y.T]
            binary_columns = all(len(col) <= 2 for col in uniques_per_col)  # noqa: PLR2004
            if binary_columns:
                warnings.warn(
                    f"`{task_hint=}` but `{inferred_type=}`."
                    " Set to `multilabel-indicator` as <=2 unique values per column."
                    " To silence this, provide a specific task type to"
                    f"`task_hint=` from {_valid_task_types}.",
                    MismatchedTaskTypeWarning,
                    stacklevel=2,
                )
                return "multilabel-indicator"
            else:  # noqa: RET505
                warnings.warn(
                    f"`{task_hint=}` but `{inferred_type=}`."
                    " Set to `multiclass-multioutput` as at least one column has"
                    " >2 unique values."
                    " To silence this, provide a specific task type to"
                    f"`task_hint=` from {_valid_task_types}.",
                    MismatchedTaskTypeWarning,
                    stacklevel=2,
                )
                return "multiclass-multioutput"
        case _:
            raise RuntimeError(
                f"Unreachable, please report this bug. {task_hint=}, {inferred_type=}",
            )


def _fit(
    estimator: BaseEstimatorT,
    X: pd.DataFrame | np.ndarray,  # noqa: N803
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    i_train: np.ndarray,
    profiler: Profiler,
    fit_params: Mapping[str, Any],
    scorer_params: Mapping[str, Any],
    scorers: Mapping[str, _Scorer] | None = None,
) -> tuple[BaseEstimatorT, Mapping[str, float] | None]:
    _fit_params = _check_method_params(X, params=fit_params, indices=i_train)
    _scorer_params_train = _check_method_params(X, scorer_params, indices=i_train)

    X_train, y_train = _safe_split(estimator, X, y, indices=i_train)
    with profiler("fit"):
        if y_train is None:
            estimator.fit(X_train, **_fit_params)  # type: ignore
        else:
            estimator.fit(X_train, y_train, **_fit_params)  # type: ignore

    train_scores = None
    if scorers is not None:
        train_scores = _score(
            estimator=estimator,
            X_test=X_train,
            y_test=y_train,
            scorer=scorers,
            score_params=_scorer_params_train,
            error_score="raise",
        )
        assert isinstance(train_scores, dict)
        for k, v in train_scores.items():
            # Can return list or np.bool_
            # We do not want a list to pass (i.e. if [x] shouldn't pass if check)
            # Also, we can't use `np.bool_` is `True` as `np.bool_(True) is not True`.
            # Hence we have to use equality checking
            # God I feel like I'm doing javascript
            if np.isfinite(v) != True:  # noqa: E712
                raise ValueError(
                    f"Scorer {k} returned {v} for train fold. The scorer should"
                    " should return a finite float",
                )

    return estimator, train_scores


def _score_val_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,  # noqa: N803
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    i_train: np.ndarray,
    i_val: np.ndarray,
    scorer_params: Mapping[str, Any],
    scorers: Mapping[str, _Scorer],
    profiler: Profiler,
) -> Mapping[str, float]:
    _scorer_params_test = _check_method_params(X, scorer_params, indices=i_val)
    X_t, y_t = _safe_split(estimator, X, y, indices=i_val, train_indices=i_train)

    with profiler("score"):
        scores = _score(
            estimator=estimator,
            X_test=X_t,
            y_test=y_t,
            scorer=scorers,
            score_params=_scorer_params_test,
            error_score="raise",
        )
        # NOTE: Despite `error_score="raise"`, this will not raise
        # for `inf` or `nan` values.
        assert isinstance(scores, dict)
        for k, v in scores.items():
            # Can return list or np.bool_
            # We do not want a list to pass (i.e. if [x] shouldn't pass if check)
            # Also, we can't use `np.bool_` is `True` as `np.bool_(True) is not True`.
            # Hence we have to use equality checking
            # God I feel like I'm doing javascript
            if np.isfinite(v) != True:  # noqa: E712
                raise ValueError(
                    f"Scorer {k} returned {v} for validation fold. The scorer"
                    " should return a finite float",
                )

        return scores


def _evaluate_fold(
    estimator: BaseEstimatorT,
    X: pd.DataFrame | np.ndarray,  # noqa: N803
    y: pd.Series | pd.DataFrame | np.ndarray,
    *,
    i_train: np.ndarray,
    i_val: np.ndarray,
    profiler: Profiler,
    scorers: Mapping[str, _Scorer],
    fit_params: Mapping[str, Any],
    scorer_params: Mapping[str, Any],
    train_score: bool,
) -> tuple[BaseEstimatorT, Mapping[str, float], Mapping[str, float] | None]:
    # These return new dictionaries

    fitted_estimator, train_scores = _fit(
        estimator=estimator,
        X=X,
        y=y,
        i_train=i_train,
        profiler=profiler,
        fit_params=fit_params,
        scorer_params=scorer_params,
        scorers=scorers if train_score else None,
    )

    val_scores = _score_val_fold(
        estimator=fitted_estimator,
        X=X,
        y=y,
        i_train=i_train,
        i_val=i_val,
        scorers=scorers,
        scorer_params=scorer_params,
        profiler=profiler,
    )

    return fitted_estimator, val_scores, train_scores


def _iter_cross_validate(
    estimator: BaseEstimatorT,
    X: Stored[pd.DataFrame | np.ndarray],  # noqa: N803
    y: Stored[pd.Series | pd.DataFrame | np.ndarray],
    splitter: BaseShuffleSplit | BaseCrossValidator,
    scorers: Mapping[str, _Scorer],
    *,
    params: Mapping[str, Any | Stored[Any]] | None = None,
    profiler: Profiler | None = None,
    train_score: bool = False,
) -> Iterator[tuple[BaseEstimatorT, Mapping[str, float], Mapping[str, float] | None]]:
    profiler = Profiler(disabled=True) if profiler is None else profiler
    params = {} if params is None else params
    loaded_params: dict[str, Any] = {
        k: v.load() if isinstance(v, Stored) else v for k, v in params.items()
    }

    # NOTE: This flow adapted from sklearns 1.4 cross_validate
    _scorer = _MultimetricScorer(scorers=scorers, raise_exc=True)
    routed_params = _route_params(splitter, estimator, _scorer, **loaded_params)
    fit_params = routed_params["estimator"]["fit"]
    scorer_params = routed_params["scorer"]["score"]

    # Notably, this is an iterator
    X_loaded = X.load()
    y_loaded = y.load()
    indicies = splitter.split(X_loaded, y_loaded, **routed_params["splitter"]["split"])

    fit_params = fit_params if fit_params is not None else {}
    scorer_params = scorer_params if scorer_params is not None else {}

    for i_train, i_test in indicies:
        # Sadly this function needs the full X and y due to its internal checks
        yield _evaluate_fold(
            estimator=estimator,
            X=X_loaded,
            y=y_loaded,
            i_train=i_train,
            i_val=i_test,
            profiler=profiler,
            scorers=scorers,
            fit_params=fit_params,
            scorer_params=scorer_params,
            train_score=train_score,
        )


def cross_validate_task(  # noqa: D103, PLR0913, C901, PLR0912
    trial: Trial,
    pipeline: Node,
    *,
    X: Stored[np.ndarray | pd.DataFrame],  # noqa: N803
    y: Stored[np.ndarray | pd.Series | pd.DataFrame],
    splitter: BaseShuffleSplit | BaseCrossValidator,
    additional_scorers: Mapping[str, _Scorer] | None,
    train_score: bool = False,
    store_models: bool = True,
    params: Mapping[str, Stored[Any] | Any] | None = None,
    builder: Literal["sklearn"] | Callable[[Node], BaseEstimator] = "sklearn",
    build_params: Mapping[str, Any] | None = None,
    on_error: Literal["fail", "raise"] = "fail",
) -> Trial.Report:
    params = {} if params is None else params
    # Make sure to load all the stored values

    build_params = {} if build_params is None else build_params
    random_state = amltk.randomness.as_randomstate(trial.seed)

    # TODO: Could possibly include `transform_context` here to `configure()`
    estimator = pipeline.configure(
        trial.config,
        params={"random_state": random_state},
    ).build(builder, **build_params)

    scorers: dict[str, _Scorer] = {}
    for metric_name, metric in trial.metrics.items():
        match metric.fn:
            case None:
                try:
                    scorer = get_scorer(metric_name)
                    scorers[metric_name] = scorer
                except ValueError as e:
                    raise ValueError(
                        f"Could not find scorer for {metric_name=} in sklearn."
                        " Please provide one with `Metric(fn=...)` or a valid"
                        " name that can be used with sklearn's `get_scorer`",
                    ) from e
            case _Scorer():  # type: ignore
                scorers[metric_name] = metric
            case _:
                # We do a best effort here and try to convert the metric to
                # an sklearn scorer.
                warnings.warn(
                    f"Found a metric with a custom function for {metric_name=}."
                    " Attempting to convert it to an sklearn scorer. This may"
                    " fail. If it does, please first your function to an sklearn"
                    " scorer with `sklearn.metrics.make_scorer` and then pass"
                    " it to `Metric(fn=...)`",
                    ImplicitMetricConversionWarning,
                    stacklevel=2,
                )
                # This may fail
                scorers[metric_name] = metric.as_scorer()

    if additional_scorers is not None:
        scorers.update(additional_scorers)

    cv_iter = _iter_cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        splitter=splitter,
        scorers=scorers,
        params=params,
        profiler=trial.profiler,
        train_score=train_score,
    )
    all_train_scores: dict[str, list[float]] = defaultdict(list)
    all_val_scores: dict[str, list[float]] = defaultdict(list)

    try:
        # Main cv loop
        for i, (_trained_estimator, _val_scores, _train_scores) in trial.profiler.each(
            enumerate(cv_iter),
            name="cv",
            itr_name="fold",
        ):
            if store_models:
                trial.store({f"model_{i}.pkl": _trained_estimator})

            trial.summary.update({f"fold_{i}:{k}": v for k, v in _val_scores.items()})
            for k, v in _val_scores.items():
                all_val_scores[k].append(v)

            if _train_scores is not None:
                trial.summary.update(
                    {f"fold_{i}:train_{k}": v for k, v in _train_scores.items()},
                )
                for k, v in _train_scores.items():
                    all_train_scores[k].append(v)

    except Exception as e:  # noqa: BLE001
        trial.dump_exception(e)
        if on_error == "raise":
            raise e
        return trial.fail(e)
    else:
        mean_val_scores = {k: np.mean(v) for k, v in all_val_scores.items()}
        std_val_scores = {k: np.std(v) for k, v in all_val_scores.items()}
        trial.summary.update({f"mean_{k}": v for k, v in mean_val_scores.items()})
        trial.summary.update({f"std_{k}": v for k, v in std_val_scores.items()})

        if any(all_train_scores):
            mean_train_scores = {k: np.mean(v) for k, v in all_train_scores.items()}
            std_train_scores = {k: np.std(v) for k, v in all_train_scores.items()}
            trial.summary.update(
                {f"train_mean_{k}": v for k, v in mean_train_scores.items()},
            )
            trial.summary.update(
                {f"train_std_{k}": v for k, v in std_train_scores.items()},
            )

        metrics_to_report = {
            k: float(v) for k, v in mean_val_scores.items() if k in trial.metrics
        }
        return trial.success(**metrics_to_report)


class CVEvaluation(EvaluationProtocol):
    """Cross-validation evaluation protocol.

    This protocol will create a cross-validation task to be used in parallel and
    optimization. It represents a typical cross-validation evaluation for sklearn.

    Aside from the init parameters, it expects:
    * The pipeline you are optimizing can be made into a [sklearn.pipeline.Pipeline][]
    calling [`.build("sklearn")`][amltk.pipeline.Node.build].
    * The seed for the trial will be passed as a param to
    [`.configure()`][amltk.pipeline.Node.configure]. If you have a component
    that accepts a `random_state` parameter, you can use a
    [`request()`][amltk.pipeline.request] so that it will be seeded correctly.

    ```python exec="true" source="material-block" result="python"
    from amltk.sklearn import CVEvaluation
    from amltk.pipeline import Component, request
    from amltk.optimization import Metric

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import get_scorer
    from sklearn.datasets import load_iris
    from pathlib import Path

    pipeline = Component(
        RandomForestClassifier,
        config={"random_state": request("random_state")},
        space={"n_estimators": (10, 100), "criterion": ["gini", "entropy"]},
    )

    working_dir = Path("./some-path")
    X, y = load_iris(return_X_y=True)
    evaluator = CVEvaluation(
        X,
        y,
        n_splits=3,
        strategy="cv",
        additional_scorers={"roc_auc": get_scorer("roc_auc_ovr")},
        store_models=False,
        train_score=True,
        working_dir=working_dir,
    )

    history = pipeline.optimize(
        target=evaluator,
        metric=Metric("accuracy", minimize=False, bounds=(0, 1)),
        working_dir=working_dir,
    )
    print(history.df())
    evaluator.bucket.rmdir()  # Cleanup
    ```
    """

    TMP_DIR_PREFIX: ClassVar[str] = "amltk-sklearn-cv-evaluation-data-"
    """Prefix for temporary directory names.

    This is only used when `working_dir` is not specified. If not specified
    you can control the tmp dir location by setting the `TMPDIR`
    environment variable. By default this is `/tmp`.

    When using a temporary directory, it will be deleted by default,
    controlled by the `delete_working_dir=` argument.
    """

    _X_FILENAME: ClassVar[str] = "X.pkl"
    """The name of the file to store the features in."""

    _Y_FILENAME: ClassVar[str] = "y.pkl"
    """The name of the file to store the target in."""

    _PARAM_EXTENSION_MAPPING: ClassVar[dict[type[Sized], str]] = {
        np.ndarray: "npy",
        pd.DataFrame: "pdpickle",
        pd.Series: "pdpickle",
    }
    """The mapping from types to extensions in
    [`params`][amltk.sklearn.evaluation.CVEvaluation.params].

    If the parameter is an instance of one of these types, and is larger than
    [`LARGE_PARAM_HEURISTIC`][amltk.sklearn.evaluation.CVEvaluation.LARGE_PARAM_HEURISTIC],
    then it will be stored to disk and loaded back up in the task.
    """

    LARGE_PARAM_HEURISTIC: ClassVar[int] = 100
    """The number of parameters that is considered large and will be stored to disk.

    When launching tasks, pickling and streaming large data to tasks can be expensive.
    This parameter checks if the object is large and if so, stores it to disk and
    gives it to the task as a [`Stored`][amltk.store.stored.Stored] object instead
    """

    task_type: TaskTypeName
    """The inferred task type."""

    additional_scorers: Mapping[str, _Scorer] | None
    """Additional scorers that will be used."""

    bucket: PathBucket
    """The bucket to use for storing data.

    For cleanup, you can call
    [`bucket.rmdir()`][amltk.store.paths.path_bucket.PathBucket.rmdir].
    """

    splitter: BaseShuffleSplit | BaseCrossValidator
    """The splitter that will be used."""

    params: Mapping[str, Any | Stored[Any]]
    """Parameters to pass to the estimator, splitter or scorers.

    Please see https://scikit-learn.org/stable/metadata_routing.html for
    more.
    """

    store_models: bool
    """Whether models will be stored in the trial."""

    train_score: bool
    """Whether scores will be calculated on the training data as well."""

    X_stored: Stored[np.ndarray | pd.DataFrame]
    """The stored features.

    You can call [`.load()`][amltk.store.stored.Stored.load] to load the
    data.
    """

    y_stored: Stored[np.ndarray | pd.Series | pd.DataFrame]
    """The stored target.

    You can call [`.load()`][amltk.store.stored.Stored.load] to load the
    data.
    """

    def __init__(  # noqa: PLR0913
        self,
        X: pd.DataFrame | np.ndarray,  # noqa: N803
        y: pd.Series | pd.DataFrame | np.ndarray,
        *,
        strategy: (
            Literal["holdout", "cv"] | BaseShuffleSplit | BaseCrossValidator
        ) = "cv",
        n_splits: int = 5,  # sklearn default
        holdout_size: float = 0.33,
        train_score: bool = False,
        store_models: bool = False,
        additional_scorers: Mapping[str, _Scorer] | None = None,
        random_state: Seed | None = None,  # Only used if cv is an int/float
        params: Mapping[str, Any] | None = None,
        task_hint: (
            TaskTypeName | Literal["classification", "regression", "auto"]
        ) = "auto",
        working_dir: str | Path | PathBucket | None = None,
        on_error: Literal["raise", "fail"] = "fail",
    ) -> None:
        """Initialize the evaluation protocol.

        Args:
            X: The features to use for training.
            y: The target to use for training.
            strategy: The cross-validation strategy to use. This can be either
                `#!python "holdout"` or `#!python "cv"`. Please see the related
                arguments below. If a scikit-learn cross-validator is provided,
                this will be used directly.
            n_splits: The number of cross-validation splits to use.
                This argument will be ignored if `#!python strategy="holdout"`
                or a custom splitter is provided for `strategy=`.
            holdout_size: The size of the holdout set to use. This argument
                will be ignored if `#!python strategy="cv"` or a custom splitter
                is provided for `strategy=`.
            train_score: Whether to score on the training data as well. This
                will take extra time as predictions will be made on the
                training data as well.
            store_models: Whether to store the trained models in the trial.
            additional_scorers: Additional scorers to use.
            random_state: The random state to use for the cross-validation
                `strategy=`. If a custom splitter is provided, this will be
                ignored.
            params: Parameters to pass to the estimator, splitter or scorers.
                See https://scikit-learn.org/stable/metadata_routing.html for
                more information.
            task_hint: A string indicating the task type matching those
                use by sklearn's `type_of_target`. This can be either
                `#!python "binary"`, `#!python "multiclass"`,
                `#!python "multilabel-indicator"`, `#!python "continuous"`,
                `#!python "continuous-multioutput"` or
                `#!python "multiclass-multioutput"`.

                You can also provide `#!python "classification"` or
                `#!python "regression"` for a more general hint.

                If not provided, this will be inferred from the target data.
                If you know this value, it is recommended to provide it as
                sometimes the target is ambiguous and sklearn may infer
                incorrectly.
            working_dir: The directory to use for storing data. If not provided,
                a temporary directory will be used. If provided as a string
                or a `Path`, it will be used as the path to the directory.
            on_error: What to do if an error occurs in the task. This can be
                either `#!python "raise"` or `#!python "fail"`. If `#!python "raise"`,
                the error will be raised and the task will fail. If `#!python "fail"`,
                the error will be caught and the task will report a failure report
                with the error message stored inside.
                Set this to `#!python "fail"` if you want to continue optimization
                even if some trials fail.
        """
        super().__init__()
        match working_dir:
            case None:
                tmpdir = Path(
                    tempfile.mkdtemp(
                        prefix=self.TMP_DIR_PREFIX,
                        suffix=datetime.now().isoformat(),
                    ),
                )
                bucket = PathBucket(tmpdir)
            case str() | Path():
                bucket = PathBucket(working_dir)
            case PathBucket():
                bucket = working_dir

        match task_hint:
            case "classification" | "regression" | "auto":
                task_type = identify_task_type(y, task_hint=task_hint)
            case (
                "binary"
                | "multiclass"
                | "multilabel-indicator"
                | "continuous"
                | "continuous-multioutput"
                | "multiclass-multioutput"
            ):
                task_type = task_hint
            case _:
                raise ValueError(
                    f"Invalid {task_hint=} provided. Must be in {_valid_task_types}",
                )

        match strategy:
            case "cv":
                splitter = _default_cv_resampler(
                    task_type,
                    n_splits=n_splits,
                    random_state=random_state,
                )
            case "holdout":
                splitter = _default_holdout(
                    task_type,
                    holdout_size=holdout_size,
                    random_state=random_state,
                )
            case _:
                splitter = strategy

        self.task_type = task_type
        self.additional_scorers = additional_scorers
        self.bucket = bucket
        self.splitter = splitter
        self.params = dict(params) if params is not None else {}
        self.store_models = store_models
        self.train_score = train_score

        self.X_stored = self.bucket[self._X_FILENAME].put(X)
        self.y_stored = self.bucket[self._Y_FILENAME].put(y)

        # We apply a heuristic that "large" parameters, such as sample_weights
        # should be stored to disk as transferring them directly to subprocess as
        # parameters is quite expensive (they must be non-optimally pickled and
        # streamed to the receiving process). By saving it to a file, we can
        # make use of things like numpy/pandas specific efficient pickling
        # protocols and also avoid the need to stream it to the subprocess.
        storable_params = {
            k: v
            for k, v in self.params.items()
            if hasattr(v, "__len__") and len(v) > self.LARGE_PARAM_HEURISTIC  # type: ignore
        }
        for k, v in storable_params.items():
            match subclass_map(v, self._PARAM_EXTENSION_MAPPING, default=None):  # type: ignore
                case (_, extension_to_save_as):
                    ext = extension_to_save_as
                case _:
                    ext = "pkl"

            self.params[k] = self.bucket[f"{k}.{ext}"].put(v)

        # This is the actual function that will be called in the task
        self.fn = partial(
            cross_validate_task,
            X=self.X_stored,
            y=self.y_stored,
            splitter=self.splitter,
            additional_scorers=self.additional_scorers,
            params=self.params,
            store_models=self.store_models,
            train_score=self.train_score,
            builder="sklearn",  # TODO: Allow user to specify? e.g. custom builder
            build_params=None,  # TODO: Allow user to specify? e.g. Imblearn pipeline
            on_error=on_error,
        )

    @override
    def task(
        self,
        scheduler: Scheduler,
        plugins: Plugin | Iterable[Plugin] | None = None,
    ) -> Task[[Trial, Node], Trial.Report]:
        return scheduler.task(self.fn, plugins=plugins if plugins is not None else ())
