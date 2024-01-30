"""TODO upon review."""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, TypeVar
from typing_extensions import override

import numpy as np
import pandas as pd
from sklearn import clone
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
    "continuous",
    "continuous-multioutput",
    "multiclass-multioutput",
]


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


def _default_resampler(
    task_type: TaskTypeName,
    *,
    n_splits: int,
    random_state: Seed | None = None,
    train_size: float = 0.67,
) -> BaseShuffleSplit | BaseCrossValidator:
    if n_splits < 1:
        raise ValueError("Must have at least one split")

    is_holdout = n_splits == 1
    random_state = amltk.randomness.as_int(random_state)

    # TODO: Custom splitter for edge case where we only have one label
    # For a given class
    # if "The least populated class in y has only" in e.args[0]:
    # Required for both of them here
    match task_type:
        case "binary" | "multiclass":
            if is_holdout:
                return StratifiedShuffleSplit(
                    n_splits=1,
                    random_state=random_state,
                    train_size=train_size,
                )
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        # NOTE: They don't natively support multilabel-indicator for stratified
        case "multilabel-indicator" | "multiclass-multioutput":
            if is_holdout:
                return ShuffleSplit(
                    n_splits=1,
                    random_state=random_state,
                    train_size=train_size,
                )

            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        case "continuous" | "continuous-multioutput":
            if is_holdout:
                return ShuffleSplit(
                    n_splits=1,
                    random_state=random_state,
                    train_size=train_size,
                )

            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        case _:
            raise ValueError(f"Don't know how to handle {task_type=} with {n_splits=}")


def identify_task_type(  # noqa: PLR0911
    y: np.ndarray | pd.Series | pd.DataFrame,
    *,
    is_classification: bool | None = None,
) -> TaskTypeName:
    """Identify the task type from the target data."""
    sklearn_target_type = type_of_target(y)
    match is_classification:
        case None:
            return sklearn_target_type
        case True:
            match sklearn_target_type:
                case "continuous":
                    unique_values = np.unique(y)
                    if len(unique_values) == 2:  # noqa: PLR2004
                        return "binary"
                    return "multiclass"
                case "continuous-multioutput":
                    return "multiclass-multioutput"
                case _:
                    return sklearn_target_type
        case False:
            match sklearn_target_type:
                case "binary" | "multiclass":
                    return "continuous"
                case "multiclass-multioutput" | "multilabel-indicator":
                    return "continuous-multioutput"
                case _:
                    return sklearn_target_type


def _iter_cross_validate(
    estimator: BaseEstimatorT,
    X: pd.DataFrame | np.ndarray,  # noqa: N803
    y: pd.Series | pd.DataFrame | np.ndarray,
    splitter: BaseShuffleSplit | BaseCrossValidator,
    scorers: Mapping[str, _Scorer],
    *,
    params: Mapping[str, Any] | None = None,
    profiler: Profiler | None = None,
    train_score: bool = False,
) -> Iterator[tuple[BaseEstimatorT, Mapping[str, float], Mapping[str, float] | None]]:
    # NOTE: This is adapted from sklearns 1.4 cross_validate

    profiler = Profiler(disabled=True) if profiler is None else profiler
    _scorer = _MultimetricScorer(scorers=scorers, raise_exc=True)

    params = {} if params is None else params
    routed_params = _route_params(splitter, estimator, _scorer, **params)
    fit_params = routed_params["estimator"]["fit"]
    scorer_params = routed_params["scorer"]["score"]

    # Notably, this is an iterator
    indicies = splitter.split(X, y, **routed_params["splitter"]["split"])

    fit_params = fit_params if fit_params is not None else {}
    scorer_params = scorer_params if scorer_params is not None else {}

    for i_train, i_test in indicies:
        # These return new dictionaries
        _fit_params = _check_method_params(X, params=fit_params, indices=i_train)
        _scorer_params_train = _check_method_params(X, scorer_params, indices=i_train)
        _scorer_params_test = _check_method_params(X, scorer_params, indices=i_test)

        X_train, y_train = _safe_split(estimator, X, y, indices=i_train)

        with profiler("fit"):
            new_estimator: BaseEstimatorT
            new_estimator = clone(estimator)  # type: ignore
            if y_train is None:
                new_estimator.fit(X_train, **_fit_params)  # type: ignore
            else:
                new_estimator.fit(X_train, y_train, **_fit_params)  # type: ignore

        X_t, y_t = _safe_split(estimator, X, y, indices=i_test, train_indices=i_train)

        with profiler("score"):
            scores = _score(
                estimator=new_estimator,
                X_test=X_t,
                y_test=y_t,
                scorer=scorers,
                score_params=_scorer_params_test,
                error_score="raise",
            )
            assert isinstance(scores, dict)

        if train_score is True:
            train_scores = _score(
                estimator=new_estimator,
                X_test=X_train,
                y_test=y_train,
                scorer=scorers,
                score_params=_scorer_params_train,
                error_score="raise",
            )
            assert isinstance(train_scores, dict)
        else:
            train_scores = {}

        yield new_estimator, scores, train_scores


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
) -> Trial.Report:
    params = {} if params is None else params
    # Make sure to load all the stored values

    loaded_params: dict[str, Any] = {
        k: v.load() if isinstance(v, Stored) else v for k, v in params.items()
    }
    build_params = {} if build_params is None else build_params
    # TODO: Could possibly include `transform_context` here
    estimator = (
        pipeline.configure(trial.config, params={"random_state": trial.seed})
        .build(builder, **build_params)
        # TODO: Hook to allow user to do this?
        .set_output(transform="pandas")  # type: ignore
    )

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
                raise ValueError(
                    "Cannot use a metric with a function that is not a"
                    " `sklearn.metrics._Scorer`.",
                )

    if additional_scorers is not None:
        scorers.update(additional_scorers)

    cv_iter = _iter_cross_validate(
        estimator=estimator,
        X=X.load(),
        y=y.load(),
        splitter=splitter,
        scorers=scorers,
        params=loaded_params,
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

    pipeline = Component(
        RandomForestClassifier,
        config={"random_state": request("random_state")},
        space={"n_estimators": (10, 100), "critera": ["gini", "entropy"]},
    )
    evaluator = CVEvaluation(
        X,
        y,
        cv=3,
        additional_scorers={"f1": get_scorer("f1")},
        store_models=False,
        train_score=True,
    )

    history = pipeline.optimize(
        target=evaluator,
        metrics=Metric("accuracy", minimize=False, bounds=(0, 1)),
        n_workers=4,
    )
    print(history.df())
    ```
    """

    TMP_DIR_PREFIX: ClassVar[str] = "amltk-sklearn-cv-evaluation-data-"
    """Prefix for temporary directory names.

    This is only used when `datadir` is not specified. If not specified
    you can control the tmp dir location by setting the `TMPDIR`
    environment variable. By default this is `/tmp`.

    When using a temporary directory, it will be deleted by default,
    controlled by the `delete_datadir=` argument.
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

    databucket: PathBucket
    """The bucket to use for storing data.

    For cleanup, you can call
    [`databucket.rmdir()`][amltk.store.paths.path_bucket.PathBucket.rmdir].
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
        cv: int | float | BaseShuffleSplit | BaseCrossValidator,
        train_score: bool = False,
        store_models: bool = True,
        additional_scorers: Mapping[str, _Scorer] | None = None,
        random_state: Seed | None = None,  # Only used if cv is an int/float
        params: Mapping[str, Any] | None = None,
        is_classification: bool | None = None,
        datadir: str | Path | PathBucket | None = None,
    ) -> None:
        """Initialize the evaluation protocol.

        Args:
            X: The features to use for training.
            y: The target to use for training.
            cv: The cross-validation strategy to use. This can be either an
                integer, a float, or a scikit-learn cross-validator.
                If an integer is provided, it will be used as the number of
                folds to use. If a float is provided, it will be used as the
                train size in a train/test split.
                If a scikit-learn cross-validator is provided, this will be
                used directly.
            train_score: Whether to score on the training data as well. This
                will take extra time as predictions will be made on the
                training data as well.
            store_models: Whether to store the trained models in the trial.
            additional_scorers: Additional scorers to use.
            random_state: The random state to use for the cross-validation
                strategy. This is only used if `cv` is an integer or a float.
            params: Parameters to pass to the estimator, splitter or scorers.
                See https://scikit-learn.org/stable/metadata_routing.html for
                more information.
            is_classification: Whether the task is a classification task.
                If not provided, this will be inferred from the target data.
                If you know this value, it is recommended to provide it as
                sometimes the target is ambiguous and sklearn may infer
                incorrectly.
            datadir: The directory to use for storing data. If not provided,
                a temporary directory will be used. If provided as a string
                or a `Path`, it will be used as the path to the directory.
        """
        super().__init__()
        match datadir:
            case None:
                tmpdir = tempfile.TemporaryDirectory(prefix=self.TMP_DIR_PREFIX)
                databucket = PathBucket(tmpdir.name)
            case str() | Path():
                databucket = PathBucket(datadir)
            case PathBucket():
                databucket = datadir

        task_type = identify_task_type(y, is_classification=is_classification)
        match cv:
            case int():
                if cv <= 1:
                    raise ValueError("cv must be > 1 if provided as an int")
                cv = _default_resampler(
                    task_type,
                    n_splits=cv,
                    random_state=random_state,
                )
            case float():
                if not 0 < cv < 1:
                    raise ValueError("cv must be > 0 and < 1 if provided as a float")
                cv = _default_resampler(
                    task_type,
                    n_splits=1,
                    random_state=random_state,
                    train_size=cv,
                )
            case _:
                pass

        self.task_type = task_type
        self.additional_scorers = additional_scorers
        self.databucket = databucket
        self.is_classification = is_classification
        self.splitter = cv
        self.params = dict(params) if params is not None else {}
        self.store_models = store_models
        self.train_score = train_score

        self.X_stored = self.databucket[self._X_FILENAME].put(X)
        self.y_stored = self.databucket[self._Y_FILENAME].put(y)

        # We apply a heuristic that "large" parameters, such as sample_weights
        # should be stored to disk as transfering them directly to subprocess as
        # parameters is quite expensive (they must be non-optimally pickled and
        # streamed to the recieving process). By saving it to a file, we can
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

            self.params[k] = self.databucket[f"{k}.{ext}"].put(v)

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
        )

    @override
    def task(
        self,
        scheduler: Scheduler,
        plugins: Plugin | Iterable[Plugin] | None = None,
    ) -> Task[[Trial, Node], Trial.Report]:
        return scheduler.task(self.fn, plugins=plugins if plugins is not None else ())
