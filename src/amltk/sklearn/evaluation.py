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
from asyncio import Future
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sized
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)
from typing_extensions import override

import numpy as np
import pandas as pd
from more_itertools import all_equal
from sklearn.base import BaseEstimator, clone
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
from sklearn.utils import Bunch
from sklearn.utils._metadata_requests import _routing_enabled
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
    CVEarlyStoppedError,
    ImplicitMetricConversionWarning,
    MismatchedTaskTypeWarning,
    TrialError,
)
from amltk.optimization import Trial
from amltk.profiling.profiler import Profiler
from amltk.scheduling import Plugin, Task
from amltk.scheduling.events import Emitter, Event
from amltk.scheduling.plugins.comm import Comm
from amltk.store import Stored
from amltk.store.paths.path_bucket import PathBucket

if TYPE_CHECKING:
    from sklearn.model_selection import (
        BaseCrossValidator,
        BaseShuffleSplit,
    )

    from amltk.pipeline import Node
    from amltk.randomness import Seed

P = ParamSpec("P")
R = TypeVar("R")


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
"""A type alias for the task type name as defined by sklearn."""

_valid_task_types: tuple[TaskTypeName, ...] = (
    "binary",
    "multiclass",
    "multilabel-indicator",
    "multiclass-multioutput",
    "continuous",
    "continuous-multioutput",
)

XLike: TypeAlias = pd.DataFrame | np.ndarray
"""A type alias for X input data type as defined by sklearn."""

YLike: TypeAlias = pd.Series | pd.DataFrame | np.ndarray
"""A type alias for y input data type as defined by sklearn."""

PostSplitSignature: TypeAlias = Callable[
    [Trial, int, "CVEvaluation.PostSplitInfo"],
    "CVEvaluation.PostSplitInfo",
]
"""A type alias for the post split callback signature.

Please see [`PostSplitInfo`][amltk.sklearn.evaluation.CVEvaluation.PostSplitInfo]
for more information on the information available to this callback.

```python
def my_post_split(
    trial: Trial,
    split_number: int,
    eval: CVEvalauation.PostSplitInfo
) -> CVEvaluation.PostSplitInfo:
    ...
```
"""


def _check_valid_scores(
    scores: Mapping[str, float] | Number,
    split: str,
) -> Mapping[str, float]:
    assert isinstance(scores, Mapping)
    for k, v in scores.items():
        # Can return list or np.bool_
        # We do not want a list to pass (i.e. if [x] shouldn't pass if check)
        # We can't use `np.bool_` is `True` as `np.bool_(True) is not True`.
        # Hence we have to use equality checking
        # God I feel like I'm doing javascript
        if np.isfinite(v) != True:  # noqa: E712
            raise ValueError(
                f"Scorer {k} returned {v} for {split} split. The scorer should"
                " should return a finite float",
            )

    return scores


def _route_params(
    splitter: BaseShuffleSplit | BaseCrossValidator,
    estimator: BaseEstimator,
    _scorer: _Scorer | _MultimetricScorer,
    **params: Any,
) -> Bunch:
    if _routing_enabled():
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
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={"groups": None})
        routed_params.estimator = Bunch(fit=params)
        routed_params.scorer = Bunch(score={})
        return routed_params


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
    y: YLike,
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
    X: XLike,  # noqa: N803
    y: YLike,
    i_train: np.ndarray,
    *,
    profiler: Profiler,
    fit_params: Mapping[str, Any],
    train_score: bool,
    scorers: dict[str, _Scorer] | _MultimetricScorer,
    scorer_params: Mapping[str, Any],
) -> tuple[BaseEstimatorT, dict[str, Any], Mapping[str, float] | None, dict[str, Any]]:
    _fit_params = _check_method_params(X, params=fit_params, indices=i_train)
    X_train, y_train = _safe_split(estimator, X, y, indices=i_train)

    with profiler("fit"):
        if y_train is None:
            estimator.fit(X_train, **_fit_params)  # type: ignore
        else:
            estimator.fit(X_train, y_train, **_fit_params)  # type: ignore

    train_scores = None
    train_scorer_params: dict[str, Any] = {}
    if train_score is True:
        scorer_params_train = _check_method_params(X, scorer_params, indices=i_train)

        with profiler("train_score"):
            train_scores = _score(
                estimator=estimator,
                X_test=X_train,
                y_test=y_train,
                scorer=scorers,
                score_params=scorer_params_train,
                error_score="raise",
            )
            train_scores = _check_valid_scores(train_scores, split="train")

    return estimator, _fit_params, train_scores, train_scorer_params


def _val_score(
    fitted_estimator: BaseEstimator,
    X: XLike,  # noqa: N803
    y: YLike,
    i_train: np.ndarray,
    i_val: np.ndarray,
    *,
    profiler: Profiler,
    scorers: dict[str, _Scorer] | _MultimetricScorer,
    scorer_params: Mapping[str, Any],
) -> tuple[Mapping[str, float], dict[str, Any]]:
    scorer_params_val = _check_method_params(X, scorer_params, indices=i_val)
    X_val, y_val = _safe_split(
        fitted_estimator,
        X,
        y,
        indices=i_val,
        train_indices=i_train,
    )
    with profiler("score"):
        val_scores = _score(
            estimator=fitted_estimator,
            X_test=X_val,
            y_test=y_val,
            scorer=scorers,
            score_params=scorer_params_val,
            error_score="raise",
        )
        val_scores = _check_valid_scores(val_scores, split="val")

    return val_scores, scorer_params_val


def _evaluate_split(  # noqa: PLR0913
    estimator: BaseEstimator,
    X: XLike,  # noqa: N803
    y: YLike,
    *,
    X_test: XLike | None = None,  # noqa: N803
    y_test: YLike | None = None,
    i_train: np.ndarray,
    i_val: np.ndarray,
    profiler: Profiler,
    scorers: _MultimetricScorer,
    fit_params: Mapping[str, Any],
    scorer_params: Mapping[str, Any],
    test_scorer_params: Mapping[str, Any],
    train_score: bool,
) -> CVEvaluation.PostSplitInfo:
    # We shove all logic that requires indexing into X for train into `_fit`.
    # This is because it's easy to create an accidental copy, i.e. with _safe_split.
    # We want that memory to only exist inside that `_fit` function and to not persists
    # throughout the body here.
    fitted_estimator, fitting_params, train_scores, train_scorer_params = _fit(
        estimator=clone(estimator),  # type: ignore
        X=X,
        y=y,
        i_train=i_train,
        profiler=profiler,
        scorers=scorers,
        fit_params=fit_params,
        train_score=train_score,
        scorer_params=scorer_params,
    )

    val_scores, val_scorer_params = _val_score(
        fitted_estimator=fitted_estimator,
        X=X,
        y=y,
        i_train=i_train,
        i_val=i_val,
        profiler=profiler,
        scorers=scorers,
        scorer_params=scorer_params,
    )

    test_scores = None
    if X_test is not None and y_test is not None:
        with profiler("test_score"):
            test_scores = _score(
                estimator=fitted_estimator,
                X_test=X_test,
                y_test=y_test,
                scorer=scorers,
                score_params=test_scorer_params,
                error_score="raise",
            )
            test_scores = _check_valid_scores(test_scores, split="test")

    return CVEvaluation.PostSplitInfo(
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        i_train=i_train,
        i_val=i_val,
        model=fitted_estimator,
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        fitting_params=fitting_params,
        train_scorer_params=train_scorer_params,
        val_scorer_params=val_scorer_params,
        test_scorer_params=test_scorer_params,
    )


def _iter_cross_validate(  # noqa: PLR0913
    estimator: BaseEstimator,
    X: XLike,  # noqa: N803
    y: YLike,
    splitter: BaseShuffleSplit | BaseCrossValidator,
    scorers: Mapping[str, _Scorer],
    *,
    X_test: XLike | None = None,  # noqa: N803
    y_test: YLike | None = None,
    fit_params: Mapping[str, Any] | None = None,
    scorer_params: Mapping[str, Any] | None = None,
    splitter_params: Mapping[str, Any] | None = None,
    test_scorer_params: Mapping[str, Any] | None = None,
    profiler: Profiler | None = None,
    train_score: bool = False,
) -> Iterator[CVEvaluation.PostSplitInfo]:
    if (X_test is not None and y_test is None) or (
        y_test is not None and X_test is None
    ):
        raise ValueError(
            "Both `X_test`, `y_test` must be provided together if one is provided.",
        )

    profiler = Profiler(disabled=True) if profiler is None else profiler

    fit_params = fit_params if fit_params is not None else {}
    scorer_params = scorer_params if scorer_params is not None else {}
    splitter_params = splitter_params if splitter_params is not None else {}
    test_scorer_params = test_scorer_params if test_scorer_params is not None else {}

    for i_train, i_val in splitter.split(X, y, **splitter_params):
        # Sadly this function needs the full X and y due to its internal checks
        yield _evaluate_split(
            estimator=estimator,
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            i_train=i_train,
            i_val=i_val,
            profiler=profiler,
            scorers=_MultimetricScorer(scorers=scorers, raise_exc=True),
            fit_params=fit_params,
            scorer_params=scorer_params,
            test_scorer_params=test_scorer_params,
            train_score=train_score,
        )


def cross_validate_task(  # noqa: D103, C901, PLR0915, PLR0913
    trial: Trial,
    pipeline: Node,
    *,
    X: Stored[XLike],  # noqa: N803
    y: Stored[YLike],
    X_test: Stored[XLike] | None = None,  # noqa: N803
    y_test: Stored[YLike] | None = None,
    splitter: BaseShuffleSplit | BaseCrossValidator,
    additional_scorers: Mapping[str, _Scorer] | None,
    train_score: bool = False,
    store_models: bool = True,
    params: MutableMapping[str, Stored[Any] | Any] | None = None,
    on_error: Literal["fail", "raise"] = "fail",
    comm: Comm | None = None,
    post_split: PostSplitSignature | None = None,
    post_processing: (
        Callable[[Trial.Report, Node, CVEvaluation.CompleteEvalInfo], Trial.Report]
        | None
    ) = None,
    post_processing_requires_models: bool = False,
) -> Trial.Report:
    params = {} if params is None else params
    # Make sure to load all the stored values

    configure_params = params.pop("configure", {})
    if not isinstance(configure_params, MutableMapping):
        raise ValueError(
            f"Expected `params['configure']` to be a dict but got {configure_params=}",
        )

    if "random_state" in configure_params:
        raise ValueError(
            "You should not provide `'random_state'` in `params['configure']`"
            " as the seed is set by the optimizer.",
        )
    random_state = amltk.randomness.as_randomstate(trial.seed)
    configure_params["random_state"] = random_state

    build_params = params.pop("build", {"builder": "sklearn"})  # type: ignore
    if not isinstance(build_params, MutableMapping):
        raise ValueError(
            f"Expected `params['build']` to be a dict but got {build_params=}",
        )

    transform_context = params.pop("transform_context", None)  # type: ignore

    configured_pipeline = pipeline.configure(
        trial.config,
        transform_context=transform_context,
        params=configure_params,
    )
    estimator = configured_pipeline.build(**build_params)

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
                scorers[metric_name] = metric.fn
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

    _X = X.load()
    _y = y.load()
    _X_test = X_test.load() if X_test is not None else None
    _y_test = y_test.load() if y_test is not None else None

    n_splits = splitter.get_n_splits()
    if n_splits is None:
        raise NotImplementedError("Needs to be handled")

    loaded_params: dict[str, Any] = {
        k: v.load() if isinstance(v, Stored) else v for k, v in params.items()
    }

    # Unfortunatly there's two things that can happen here.
    # 1. The scorer requires does not require split specific param data (e.g. pos_label)
    #    * In this case, the param['pos_label'] can be used for train/val and test
    # 2. The scorer requires required split specific param data (e.g. sample_weight)
    #    * In this case, we use the split indices to select the part of
    #    `params['sample_weight']` that is required for the repsective train/val split.
    #   * This means we can not use `params['sample_weight']` for the test split, as
    #     this would require some odd hack of concatenating them and having seperate
    #     test indices passed in by the user, a pretty dumb idea.
    #
    # The easy workaround is to have the user provide `test_{key}` for something
    # like `params['test_sample_weight']`, which we can then use for the test split.
    # However this breaks the metadata routing, which introspects the objects as
    # sees that yes, indeed something has requested `sample_weight` but nothing
    # has requested `test_sample_weight`. Worse still, we would need to pass
    # `params['test_sample_weight']` to the `sample_weight=` parameter of scorer.
    #
    # Our workaround is to have users provide `test_{key}` for all the scorer params
    # which we pop into a new dict with just `{key}`, where the `test_` prefix has been
    # removed. The router will never see this dict.
    #
    # As an important caveats:
    # * We assume all keys prefixed with `test_` are scorer params.
    # * Things like `pos_label` which are split agnostic needs to be
    #   provided twice, once as `pos_label` and once as `test_pos_label`, such that
    #   the scores in test recieve th params.
    test_scorer_params = {
        k: v
        for k in list(loaded_params)
        if (v := loaded_params.pop(f"test_{k}", None)) is not None
    }

    # We've now popped out all the test params, so we can safely call
    # to `_route_params` without it complaining that nothing has requested `test_{key}`

    # NOTE: This flow adapted from sklearns 1.4 cross_validate
    # This scorer is only created for routing purposes
    multimetric_scorer = _MultimetricScorer(scorers=scorers, raise_exc=True)
    routed_params = _route_params(
        splitter=splitter,
        estimator=estimator,
        _scorer=multimetric_scorer,
        **loaded_params,
    )

    fit_params = routed_params["estimator"]["fit"]
    scorer_params = routed_params["scorer"]["score"]
    splitter_params = routed_params["splitter"]["split"]

    cv_iter = _iter_cross_validate(
        estimator=estimator,
        X=_X,
        y=_y,
        X_test=_X_test,
        y_test=_y_test,
        splitter=splitter,
        scorers=scorers,
        profiler=trial.profiler,
        train_score=train_score,
        fit_params=fit_params,
        scorer_params=scorer_params,
        splitter_params=splitter_params,
        test_scorer_params=test_scorer_params,
    )

    split_scores = CVEvaluation.SplitScores(
        val=defaultdict(list),
        train=defaultdict(list) if train_score else None,
        test=defaultdict(list) if X_test is not None else None,
    )
    models: list[BaseEstimator] | None = (
        None if not post_processing_requires_models else []
    )

    with comm.open() if comm is not None else nullcontext():
        try:
            # Open up comms if passed in, allowing for the cv early stopping mechanism
            # to communicate back to the main process
            # Main cv loop
            for i, _split_eval in trial.profiler.each(
                enumerate(cv_iter),
                name="cv",
                itr_name="split",
            ):
                if post_split is not None:
                    split_eval = post_split(trial, i, _split_eval)
                else:
                    split_eval = _split_eval

                # Update the report
                if store_models:
                    trial.store({f"model_{i}.pkl": split_eval.model})

                trial.summary.update(
                    {f"split_{i}:val_{k}": v for k, v in split_eval.val_scores.items()},
                )
                for k, v in split_eval.val_scores.items():
                    split_scores.val[k].append(v)

                if split_eval.train_scores is not None:
                    trial.summary.update(
                        {
                            f"split_{i}:train_{k}": v
                            for k, v in split_eval.train_scores.items()
                        },
                    )
                    for k, v in split_eval.train_scores.items():
                        split_scores.train[k].append(v)  # type: ignore

                if split_eval.test_scores is not None:
                    trial.summary.update(
                        {
                            f"split_{i}:test_{k}": v
                            for k, v in split_eval.test_scores.items()
                        },
                    )
                    for k, v in split_eval.test_scores.items():
                        split_scores.test[k].append(v)  # type: ignore

                if post_processing_requires_models:
                    assert models is not None
                    models.append(split_eval.model)

                # At this point, we wish to remove the split_eval object from memory
                # if possible. This doesn't actually clean up memory but marks it
                # as being viable for garbage collection.
                del split_eval

                # If there was a comm passed, we are operating under cv early stopping
                # mode, in which case we request information from the main process,
                # should we continue or stop?
                if comm is not None and i < n_splits:
                    match response := comm.request(
                        (trial, split_scores),
                        timeout=10,
                    ):
                        case True:
                            raise CVEarlyStoppedError("Early stopped!")
                        case False:
                            pass
                        case np.bool_():
                            if bool(response) is True:
                                raise CVEarlyStoppedError("Early stopped!")
                        case Exception():
                            raise response
                        case _:
                            raise RuntimeError(
                                f"Recieved {response=} which we can't handle."
                                " Please return `True`, `False` or an `Exception`"
                                f" and not a type {type(response)=}",
                            )

        except Exception as e:  # noqa: BLE001
            trial.dump_exception(e)
            report = trial.fail(e)
            if on_error == "raise":
                raise TrialError(f"Trial failed: {report}") from e

            if post_processing is not None:
                final_eval_info = CVEvaluation.CompleteEvalInfo(
                    X=_X,
                    y=_y,
                    X_test=_X_test,
                    y_test=_y_test,
                    splitter=splitter,
                    max_splits=n_splits,
                    scores=split_scores,
                    scorers=scorers,
                    models=models,
                    splitter_params=splitter_params,
                    fit_params=fit_params,
                    scorer_params=scorer_params,
                    test_scorer_params=test_scorer_params,
                )
                report = post_processing(report, pipeline, final_eval_info)

            return report
        else:
            for mname, fold_scores in split_scores.val.items():
                trial.summary[f"val_mean_{mname}"] = float(np.mean(fold_scores))
                trial.summary[f"val_std_{mname}"] = float(np.std(fold_scores))

            if split_scores.train is not None:
                for mname, fold_scores in split_scores.train.items():
                    trial.summary[f"train_mean_{mname}"] = float(np.mean(fold_scores))
                    trial.summary[f"train_std_{mname}"] = float(np.std(fold_scores))

            if split_scores.test is not None:
                for mname, fold_scores in split_scores.test.items():
                    trial.summary[f"test_mean_{mname}"] = float(np.mean(fold_scores))
                    trial.summary[f"test_std_{mname}"] = float(np.std(fold_scores))

            means_to_report = {k: trial.summary[f"val_mean_{k}"] for k in trial.metrics}
            report = trial.success(**means_to_report)

            if post_processing is not None:
                final_eval_info = CVEvaluation.CompleteEvalInfo(
                    X=_X,
                    y=_y,
                    X_test=_X_test,
                    y_test=_y_test,
                    splitter=splitter,
                    max_splits=n_splits,
                    scores=split_scores,
                    scorers=scorers,
                    models=models,
                    splitter_params=splitter_params,
                    fit_params=fit_params,
                    scorer_params=scorer_params,
                    test_scorer_params=test_scorer_params,
                )
                report = post_processing(report, pipeline, final_eval_info)

            return report


class CVEvaluation(Emitter):
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
        splitter="cv",
        additional_scorers={"roc_auc": get_scorer("roc_auc_ovr")},
        store_models=False,
        train_score=True,
        working_dir=working_dir,
    )

    history = pipeline.optimize(
        target=evaluator.fn,
        metric=Metric("accuracy", minimize=False, bounds=(0, 1)),
        working_dir=working_dir,
        max_trials=1,
    )
    print(history.df())
    evaluator.bucket.rmdir()  # Cleanup
    ```

    If you need to pass specific configuration items to your pipeline during
    configuration, you can do so using a [`request()`][amltk.pipeline.request]
    in the config of your pipeline.

    In the below example, we allow the pipeline to be configured with `"n_jobs"`
    and pass it in to the `CVEvalautor` using the `params` argument.

    ```python exec="true" source="material-block" result="python"
    from amltk.sklearn import CVEvaluation
    from amltk.pipeline import Component, request
    from amltk.optimization import Metric

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import get_scorer
    from sklearn.datasets import load_iris
    from pathlib import Path

    working_dir = Path("./some-path")
    X, y = load_iris(return_X_y=True)

    pipeline = Component(
        RandomForestClassifier,
        config={
            "random_state": request("random_state"),
            # Allow it to be configured with n_jobs
            "n_jobs": request("n_jobs", default=None)
        },
        space={"n_estimators": (10, 100), "criterion": ["gini", "entropy"]},
    )

    evaluator = CVEvaluation(
        X,
        y,
        working_dir=working_dir,
        # Use the `configure` keyword in params to pass to the `n_jobs`
        # Anything in the pipeline requesting `n_jobs` will get the value
        params={"configure": {"n_jobs": 2}}
    )
    history = pipeline.optimize(
        target=evaluator.fn,
        metric=Metric("accuracy"),
        working_dir=working_dir,
        max_trials=1,
    )
    print(history.df())
    evaluator.bucket.rmdir()  # Cleanup
    ```

    !!! tip "CV Early Stopping"

        To see more about early stopping, please see
        [`CVEvaluation.cv_early_stopping_plugin()`][amltk.sklearn.evaluation.CVEvaluation.cv_early_stopping_plugin].

    """

    SPLIT_EVALUATED: Event[[Trial, SplitScores], bool | Exception] = Event(
        "split-evaluated",
    )
    """Event that is emitted when a split has been evaluated.

    Only emitted if the evaluator plugin is being used.
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

    _X_TEST_FILENAME: ClassVar[str] = "X_test.pkl"
    """The name of the file to store the test features in."""

    _Y_FILENAME: ClassVar[str] = "y.pkl"
    """The name of the file to store the targets in."""

    _Y_TEST_FILENAME: ClassVar[str] = "y_test.pkl"
    """The name of the file to store the test targets in."""

    PARAM_EXTENSION_MAPPING: ClassVar[dict[type[Sized], str]] = {
        np.ndarray: "npy",
        pd.DataFrame: "pdpickle",
        pd.Series: "pdpickle",
    }
    """The mapping from types to extensions in
    [`params`][amltk.sklearn.evaluation.CVEvaluation.params].

    If the parameter is an instance of one of these types, and is larger than
    [`LARGE_PARAM_HEURISTIC`][amltk.sklearn.evaluation.CVEvaluation.LARGE_PARAM_HEURISTIC],
    then it will be stored to disk and loaded back up in the task.

    Please feel free to overwrite this class variable as needed.
    """

    LARGE_PARAM_HEURISTIC: ClassVar[int] = 100
    """Any item in `params=` which is greater will be stored to disk when sent to the
    worker.

    When launching tasks, pickling and streaming large data to tasks can be expensive.
    This parameter checks if the object is large and if so, stores it to disk and
    gives it to the task as a [`Stored`][amltk.store.stored.Stored] object instead.

    Please feel free to overwrite this class variable as needed.
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

    X_stored: Stored[XLike]
    """The stored features.

    You can call [`.load()`][amltk.store.stored.Stored.load] to load the
    data.
    """

    y_stored: Stored[YLike]
    """The stored target.

    You can call [`.load()`][amltk.store.stored.Stored.load] to load the
    data.
    """

    class PostSplitInfo(NamedTuple):
        """Information about the evaluation of a split.

        Attributes:
            X: The features to used for training.
            y: The targets used for training.
            X_test: The features used for testing if it was passed in.
            y_test: The targets used for testing if it was passed in.
            i_train: The train indices for this split.
            i_val: The validation indices for this split.
            model: The model that was trained in this split.
            train_scores: The training scores for this split if requested.
            val_scores: The validation scores for this split.
            test_scores: The test scores for this split if requested.
            fitting_params: Any additional fitting parameters that were used.
            train_scorer_params: Any additional scorer parameters used for evaluating
                scorers on training set.
            val_scorer_params: Any additional scorer parameters used for evaluating
                scorers on training set.
            test_scorer_params: Any additional scorer parameters used for evaluating
                scorers on training set.
        """

        X: XLike
        y: YLike
        X_test: XLike | None
        y_test: YLike | None
        i_train: np.ndarray
        i_val: np.ndarray
        model: BaseEstimator
        val_scores: Mapping[str, float]
        train_scores: Mapping[str, float] | None
        test_scores: Mapping[str, float] | None
        fitting_params: Mapping[str, Any]
        train_scorer_params: Mapping[str, Any]
        val_scorer_params: Mapping[str, Any]
        test_scorer_params: Mapping[str, Any]

    class SplitScores(NamedTuple):
        """The scores for a split.

        Attributes:
            val: The validation scores for all evaluated split.
            train: The training scores for all evaluated splits if requested.
            test: The test scores for all evaluated splits if requested.
        """

        val: Mapping[str, list[float]]
        train: Mapping[str, list[float]] | None
        test: Mapping[str, list[float]] | None

    @dataclass
    class CompleteEvalInfo:
        """Information about the final evaluation of a cross-validation task.

        This class contains information about the final evaluation of a cross-validation
        that will be passed to the post-processing function.
        """

        X: XLike
        """The features to used for training."""

        y: YLike
        """The targets used for training."""

        X_test: XLike | None
        """The features used for testing."""

        y_test: YLike | None
        """The targets used for testing."""

        splitter: BaseShuffleSplit | BaseCrossValidator
        """The splitter that was used."""

        max_splits: int
        """The maximum number of splits that were (or could have been) evaluated."""

        scores: CVEvaluation.SplitScores
        """The scores for the splits that were evaluated."""

        scorers: dict[str, _Scorer]
        """The scorers that were used."""

        models: list[BaseEstimator] | None
        """The models that were trained in each split.

        This will be `None` if `post_processing_requires_models=False`.
        """

        splitter_params: Mapping[str, Any]
        """The parameters that were used for the splitter."""

        fit_params: Mapping[str, Any]
        """The parameters that were used for fitting the estimator.

        Please use
        [`select_params()`][amltk.sklearn.evaluation.CVEvaluation.CompleteEvalInfo.select_params]
        if you need to select the params specific to a split, i.e. for `sample_weights`.
        """

        scorer_params: Mapping[str, Any]
        """The parameters that were used for scoring the estimator.

        Please use
        [`select_params()`][amltk.sklearn.evaluation.CVEvaluation.CompleteEvalInfo.select_params]
        if you need to select the params specific to a split, i.e. for `sample_weights`.
        """

        test_scorer_params: Mapping[str, Any]
        """The parameters that were used for scoring the test data.

        Please use
        [`select_params()`][amltk.sklearn.evaluation.CVEvaluation.CompleteEvalInfo.select_params]
        if you need to select the params specific to a split, i.e. for `sample_weights`.
        """

        # TODO: We don't use `test_fit_params` in our evaluator but someone could
        # potentially need it here. Fix if it becomes a problem...

        def select_params(
            self,
            params: Mapping[str, Any],
            indices: np.ndarray,
        ) -> dict[str, Any]:
            """Convinience method to select parameters for a specific split."""
            return _check_method_params(self.X, params, indices=indices)

    class _CVEarlyStoppingPlugin(Plugin):
        name: ClassVar[str] = "cv-early-stopping-plugin"

        def __init__(
            self,
            evaluator: CVEvaluation,
            *,
            strategy: CVEarlyStoppingProtocol | None = None,
            create_comms: Callable[[], tuple[Comm, Comm]] | None = None,
        ) -> None:
            super().__init__()
            self.evaluator = evaluator
            self.strategy = strategy
            self.comm_plugin = Comm.Plugin(
                create_comms=create_comms,
                parameter_name="comm",
            )

        @override
        def pre_submit(
            self,
            fn: Callable[P, R],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> tuple[Callable[P, R], tuple, dict] | None:
            return self.comm_plugin.pre_submit(fn, *args, **kwargs)

        @override
        def attach_task(self, task: Task) -> None:
            """Attach the plugin to a task.

            This method is called when the plugin is attached to a task. This
            is the place to subscribe to events on the task, create new subscribers
            for people to use or even store a reference to the task for later use.

            Args:
                task: The task the plugin is being attached to.
            """
            self.task = task
            self.comm_plugin.attach_task(task)
            task.add_event(CVEvaluation.SPLIT_EVALUATED)
            task.register(Comm.REQUEST, self._on_comm_request_ask_whether_to_continue)
            if self.strategy is not None:
                task.register(self.evaluator.SPLIT_EVALUATED, self.strategy.should_stop)
                task.register(task.RESULT, self._call_strategy_update)

        def _call_strategy_update(self, _: Future, report: Trial.Report) -> None:
            if self.strategy is not None:
                self.strategy.update(report)

        def _on_comm_request_ask_whether_to_continue(self, msg: Comm.Msg) -> None:
            if not (isinstance(msg.data, tuple) and len(msg.data) == 2):  # noqa: PLR2004
                return

            trial, scores = msg.data
            if not (
                isinstance(trial, Trial)
                and isinstance(scores, CVEvaluation.SplitScores)
            ):
                return

            non_null_responses = [
                r
                for _, r in self.task.emit(
                    self.evaluator.SPLIT_EVALUATED,
                    trial,
                    scores,
                )
                if r is not None
            ]
            logger.debug(
                f"Received responses for {self.evaluator.SPLIT_EVALUATED}:"
                f" {non_null_responses}",
            )
            match len(non_null_responses):
                case 0:
                    msg.respond(response=False)
                case 1:
                    msg.respond(response=non_null_responses[0])
                case _ if all_equal(non_null_responses):
                    msg.respond(response=non_null_responses[0])
                case _:
                    raise NotImplementedError(
                        "Multiple callbacks returned different values."
                        " Behaviour is undefined. Please aggregate behaviour"
                        " into one callback. Also please raise an issue to"
                        " discuss use cases and how to handle this.",
                    )

    def __init__(  # noqa: PLR0913
        self,
        X: XLike,  # noqa: N803
        y: YLike,
        *,
        X_test: XLike | None = None,  # noqa: N803
        y_test: YLike | None = None,
        splitter: (
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
        post_split: PostSplitSignature | None = None,
        post_processing: (
            Callable[[Trial.Report, Node, CVEvaluation.CompleteEvalInfo], Trial.Report]
            | None
        ) = None,
        post_processing_requires_models: bool = False,
    ) -> None:
        """Initialize the evaluation protocol.

        Args:
            X: The features to use for training.
            y: The target to use for training.
            X_test: The features to use for testing. If provided, all
                scorers will be calculated on this data as well.
                Must be provided with `y_test=`.

                !!! tip "Scorer params for test scoring"

                    Due to nuances of sklearn's metadata routing, if you need to provide
                    parameters to the scorer for the test data, you can prefix these
                    with `#!python "test_"`. For example, if you need to provide
                    `pos_label` to the scorer for the test data, you must provide
                    `test_pos_label` in the `params` argument.

            y_test: The target to use for testing. If provided, all
                scorers will be calculated on this data as well.
                Must be provided with `X_test=`.
            splitter: The cross-validation splitter to use. This can be either
                `#!python "holdout"` or `#!python "cv"`. Please see the related
                arguments below. If a scikit-learn cross-validator is provided,
                this will be used directly.
            n_splits: The number of cross-validation splits to use.
                This argument will be ignored if `#!python splitter="holdout"`
                or a custom splitter is provided for `splitter=`.
            holdout_size: The size of the holdout set to use. This argument
                will be ignored if `#!python splitter="cv"` or a custom splitter
                is provided for `splitter=`.
            train_score: Whether to score on the training data as well. This
                will take extra time as predictions will be made on the
                training data as well.
            store_models: Whether to store the trained models in the trial.
            additional_scorers: Additional scorers to use.
            random_state: The random state to use for the cross-validation
                `splitter=`. If a custom splitter is provided, this will be
                ignored.
            params: Parameters to pass to the estimator, splitter or scorers.
                See https://scikit-learn.org/stable/metadata_routing.html for
                more information.

                You may also additionally include the following as dictionarys:

                * `#!python "configure"`: Parameters to pass to the pipeline
                    for [`configure()`][amltk.pipeline.Node.configure]. Please
                    the example in the class docstring for more information.
                * `#!python "build"`: Parameters to pass to the pipeline for
                    [`build()`][amltk.pipeline.Node.build].

                    ```python
                    from imblearn.pipeline import Pipeline as ImbalancedPipeline
                    CVEvaluator(
                        ...,
                        params={
                            "build": {
                                "builder": "sklearn",
                                "pipeline_type": ImbalancedPipeline
                            }
                        }
                    )
                    ```

                * `#!python "transform_context"`: The transform context to use
                    for [`configure()`][amltk.pipeline.Node.configure].

                !!! tip "Scorer params for test scoring"

                    Due to nuances of sklearn's metadata routing, if you need to provide
                    parameters to the scorer for the test data, you must prefix these
                    with `#!python "test_"`. For example, if you need to provide
                    `pos_label` to the scorer for the test data, you can provide
                    `test_pos_label` in the `params` argument.

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
            post_split: If provided, this callable will be called with a
                [`PostSplitInfo`][amltk.sklearn.evaluation.CVEvaluation.PostSplitInfo].

                For example, this could be useful if you'd like to save out-of-fold
                predictions for later use.

                ```python
                def my_post_split(
                    split_number: int,
                    info: CVEvaluator.PostSplitInfo,
                ) -> None:
                    X_val, y_val = info.val
                    oof_preds = fitted_model.predict(X_val)

                    split = info.current_split
                    info.trial.store({f"oof_predictions_{split}.npy": oof_preds})
                    return info
                ```

                !!! warning "Run in the worker"

                    This callable will be pickled and sent to the worker that is
                    executing an evaluation. This means that you should mitigate
                    relying on any large objects if your callalbe is an object, as
                    the object will get pickled and sent to the worker. This also means
                    you can not rely on information obtained from other trials as when
                    sending the callable to a worker, it is no longer updatable from the
                    main process.

                    You should also avoid holding on to references to either the model
                    or large data that is passed in
                    [`PostSplitInfo`][amltk.sklearn.evaluation.CVEvaluation.PostSplitInfo]
                    to the function.

                    This parameter should primarily be used for callables that rely
                    solely on the output of the current trial and wish to store/add
                    additional information to the trial itself.

            post_processing: If provided, this callable will be called with all of the
                evaluated splits and the final report that will be returned.
                This can be used to do things such as augment the final scores
                if required, cleanup any resources or any other tasks that should be
                run after the evaluation has completed. This will be handed a
                [`Report`][amltk.optimization.trial.Trial.Report] and a
                [`CompleteEvalInfo`][amltk.sklearn.evaluation.CVEvaluation.CompleteEvalInfo],
                which contains all the information about the evaluation. If your
                function requires the individual models, you can set
                `post_processing_requires_models=True`. By default this is `False`
                as this requires having all models in memory at once.

                This can be useful when you'd like to report the score of a bagged
                model, i.e. an ensemble of all validation models. Another example
                is if you'd like to add to the summary, the score of what the model
                would be if refit on all the data.

                ```python
                from amltk.sklearn.voting import voting_with_prefitted_estimators

                # Compute the test score of all fold models bagged together
                def my_post_processing(
                    report: Trial.Report,
                    pipeline: Node,
                    info: CVEvaluator.CompleteEvalInfo,
                ) -> Trial.Report:
                    bagged_model = voting_with_prefitted_estimators(info.models)
                    acc = info.scorers["accuracy"]
                    bagged_score = acc(bagged_model, info.X_test, info.y_test)
                    report.summary["bagged_test_score"] = bagged_score
                    return report
                ```

                !!! warning "Run in the worker"

                    This callable will be pickled and sent to the worker that is
                    executing an evaluation. This means that you should mitigate
                    relying on any large objects if your callalbe is an object, as
                    the object will get pickled and sent to the worker. This also means
                    you can not rely on information obtained from other trials as when
                    sending the callable to a worker, it is no longer updatable from the
                    main process.

                    This parameter should primarily be used for callables that will
                    augment the report or what is stored with the trial. It should
                    rely solely on the current trial to prevent unexpected issues.

            post_processing_requires_models: Whether the `post_processing` function
                requires the models to be passed to it. If `True`, the models will
                be passed to the function in the `CompleteEvalInfo` object. If `False`,
                the models will not be passed to the function. By default this is
                `False` as this requires having all models in memory at once.

        """
        super().__init__()
        if (X_test is not None and y_test is None) or (
            y_test is not None and X_test is None
        ):
            raise ValueError(
                "Both `X_test`, `y_test` must be provided together if one is provided.",
            )

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
                | "multiclass-multioutput"  #
            ):
                task_type = task_hint
            case _:
                raise ValueError(
                    f"Invalid {task_hint=} provided. Must be in {_valid_task_types}"
                    f"\n{type(task_hint)=}",
                )

        match splitter:
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
                splitter = splitter  # noqa: PLW0127

        self.task_type = task_type
        self.additional_scorers = additional_scorers
        self.bucket = bucket
        self.splitter = splitter
        self.params = dict(params) if params is not None else {}
        self.store_models = store_models
        self.train_score = train_score

        self.X_stored = self.bucket[self._X_FILENAME].put(X)
        self.y_stored = self.bucket[self._Y_FILENAME].put(y)

        self.X_test_stored = None
        self.y_test_stored = None
        if X_test is not None and y_test is not None:
            self.X_test_stored = self.bucket[self._X_TEST_FILENAME].put(X_test)
            self.y_test_stored = self.bucket[self._Y_TEST_FILENAME].put(y_test)

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
            match subclass_map(v, self.PARAM_EXTENSION_MAPPING, default=None):  # type: ignore
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
            X_test=self.X_test_stored,
            y_test=self.y_test_stored,
            splitter=self.splitter,
            additional_scorers=self.additional_scorers,
            params=self.params,
            store_models=self.store_models,
            train_score=self.train_score,
            on_error=on_error,
            post_split=post_split,
            post_processing=post_processing,
            post_processing_requires_models=post_processing_requires_models,
        )

    def cv_early_stopping_plugin(
        self,
        strategy: CVEarlyStoppingProtocol
        | None = None,  # TODO: Can provide some defaults...
        *,
        create_comms: Callable[[], tuple[Comm, Comm]] | None = None,
    ) -> CVEvaluation._CVEarlyStoppingPlugin:
        """Create a plugin for a task allow for early stopping.

        ```python exec="true" source="material-block" result="python" html="true"
        from dataclasses import dataclass
        from pathlib import Path

        import sklearn.datasets
        from sklearn.tree import DecisionTreeClassifier

        from amltk.sklearn import CVEvaluation
        from amltk.pipeline import Component
        from amltk.optimization import Metric

        working_dir = Path("./some-path")
        pipeline = Component(DecisionTreeClassifier, space={"max_depth": (1, 10)})
        x, y = sklearn.datasets.load_iris(return_X_y=True)
        evaluator = CVEvaluation(x, y, n_splits=3, working_dir=working_dir)

        # Our early stopping strategy, with an `update()` and `should_stop()`
        # signature match what's expected.

        @dataclass
        class CVEarlyStopper:
            def update(self, report: Trial.Report) -> None:
                # Normally you would update w.r.t. a finished trial, such
                # as updating a moving average of the scores.
                pass

            def should_stop(self, trial: Trial, scores: CVEvaluation.SplitScores) -> bool | Exception:
                # Return True to stop, False to continue. Alternatively, return a
                # specific exception to attach to the report instead
                return True

        history = pipeline.optimize(
            target=evaluator.fn,
            metric=Metric("accuracy", minimize=False, bounds=(0, 1)),
            max_trials=1,
            working_dir=working_dir,

            # Here we insert the plugin to the task that will get created
            plugins=[evaluator.cv_early_stopping_plugin(strategy=CVEarlyStopper())],

            # Notably, we set `on_trial_exception="continue"` to not stop as
            # we expect trials to fail given the early stopping strategy
            on_trial_exception="continue",
        )
        from amltk._doc import doc_print; doc_print(print, history[0])  # markdown-exec: hide
        evaluator.bucket.rmdir()  # markdown-exec: hide
        ```

        !!! warning "Recommended settings for `CVEvaluation`

            When a trial is early stopped, it will be counted as a failed trial.
            This can conflict with the behaviour of `pipeline.optimize` which
            by default sets `on_trial_exception="raise"`, causing the optimization
            to end. If using [`pipeline.optimize`][amltk.pipeline.Node.optimize],
            to set `on_trial_exception="continue"` to continue optimization.

        This will also add a new event to the task which you can subscribe to with
        [`task.on("split-evaluated")`][amltk.sklearn.evaluation.CVEvaluation.SPLIT_EVALUATED].
        It will be passed a
        [`CVEvaluation.PostSplitInfo`][amltk.sklearn.evaluation.CVEvaluation.PostSplitInfo]
        that you can use to make a decision on whether to continue or stop. The
        passed in `strategy=` simply sets up listening to these events for you.
        You can also do this manually.

        ```python
        scores = []
        evaluator = CVEvaluation(...)
        task = scheduler.task(
            evaluator.fn,
            plugins=[evaluator.cv_early_stopping_plugin()]
        )

        @task.on("split-evaluated")
        def should_stop(trial: Trial, scores: CVEvaluation.SplitScores) -> bool | Execption:
            # Make a decision on whether to stop or continue
            return info.scores["accuracy"] < np.mean(scores.val["accuracy"])

        @task.on("result")
        def update_scores(_, report: Trial.Report) -> bool | Execption:
            if report.status is Trial.Status.SUCCESS:
                return scores.append(report.values["accuracy"])
        ```

        Args:
            strategy: The strategy to use for early stopping. Must implement the
                `update()` and `should_stop()` methods of
                [`CVEarlyStoppingProtocol`][amltk.sklearn.evaluation.CVEarlyStoppingProtocol].
                Please follow the documentation link to find out more.

                By default, when no `strategy=` is passedj this is `None` and
                this will create a [`Comm`][amltk.scheduling.plugins.comm.Comm] object,
                allowing communication between the worker running the task and the main
                process. This adds a new event to the task that you can subscribe
                to with
                [`task.on("split-evaluated")`][amltk.sklearn.evaluation.CVEvaluation.SPLIT_EVALUATED].
                This is how a passed in strategy will be called and updated.
            create_comms: A function that creates a pair of comms for the
                plugin to use. This is useful if you want to create a
                custom communication channel. If not provided, the default
                communication channel will be used.

                !!! note "Default communication channel"

                    By default we use a simple `multiprocessing.Pipe` which works
                    for parallel processses from
                    [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor].
                    This may not work if the tasks is being executed in a different
                    filesystem or depending on the executor which executes the task.

        Returns:
            The plugin to use for the task.
        """  # noqa: E501
        return CVEvaluation._CVEarlyStoppingPlugin(
            self,
            strategy=strategy,
            create_comms=create_comms,
        )


class CVEarlyStoppingProtocol(Protocol):
    """Protocol for early stopping in cross-validation.

    You class should implement the
    [`update()`][amltk.sklearn.evaluation.CVEarlyStoppingProtocol.update]
    and [`should_stop()`][amltk.sklearn.evaluation.CVEarlyStoppingProtocol.should_stop]
    methods. You can optionally inherit from this class but it is not required.

    ```python
    class MyStopper:

        def update(self, report: Trial.Report) -> None:
            if report.status is Trial.Status.SUCCESS:
                # ... do some update logic

        def should_stop(self, trial: Trial, split_infos: list[CVEvaluation.PostSplitInfo]) -> bool | Exception:
            mean_scores_up_to_current_split = np.mean([i.val_scores["accuracy"] for i in split_infos])
            if mean_scores_up_to_current_split > 0.9:
                return False  # Keep going
            else:
                return True  # Stop evaluating
    ```
    """  # noqa: E501

    def update(self, report: Trial.Report) -> None:
        """Update the protocol with a new report.

        This will be called when a trial has been completed, either successfully
        or failed. You can check for successful trials by using
        [`report.status`][amltk.optimization.Trial.Report.status].

        Args:
            report: The report from the trial.
        """
        ...

    def should_stop(
        self,
        trial: Trial,
        scores: CVEvaluation.SplitScores,
    ) -> bool | Exception:
        """Determines whether the cross-validation should stop early.

        Args:
            trial: The trial that is currently being evaluated.
            scores: The scores from the evlauated splits.

        Returns:
            `True` if the cross-validation should stop, `False` if it should
            continue, or an `Exception` if it should stop and you'd like a custom
            error to be registered with the trial.
        """
        ...
