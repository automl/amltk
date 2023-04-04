from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
from typing import List

import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from byop.ensembling.ensemble_weighting import GreedyEnsembleSelection, EnsembleWeightingCMAES
from byop.ensembling.abstract_weighted_ensemble import AbstractWeightedEnsemble
from byop.ensembling.ensemble_preprocessing import prune_base_models

# from dask_jobqueue import SLURMCluster
from byop.optimization import Trial
from byop.pipeline import Pipeline, choice, split, step
from byop.scheduling import Scheduler, Task
from byop.smac import SMACOptimizer
from byop.store import PathBucket

LEVEL = logging.INFO  # Change to DEBUG if you want the gory details shown
logger = logging.getLogger(__name__)

# TODO: Given documentation this should work, but it doesnt.
# logging.basicConfig(level=logging.INFO)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(LEVEL)

# --- Main content
pipeline = Pipeline.create(
    # Preprocessing
    split(
        "feature_preprocessing",
        (step("categoricals", SimpleImputer, space={
            "strategy": ["most_frequent", "constant"],
            "fill_value": ["missing"],
        })
         | step("ohe", OneHotEncoder,
                space={
                    "min_frequency": (0.01, 0.1),
                    "handle_unknown": ["ignore", "infrequent_if_exist"],
                },
                config={"drop": "first"},
                )
         ),
        (step("numerics", SimpleImputer, space={"strategy": ["mean", "median"]})
         | step("variance_threshold",
                VarianceThreshold,
                space={"threshold": (0.0, 0.2)}
                )
         | choice("scaler",
                  step("standard", StandardScaler),
                  step("minmax", MinMaxScaler),
                  step("robust", RobustScaler),
                  step("passthrough", FunctionTransformer),
                  )
         ),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    ),
    # Algorithm
    choice(
        "algorithm",
        step("svm", SVC, space={"C": (0.1, 10.0)}, config={"probability": True}),
        step("rf", RandomForestClassifier,
             space={
                 "n_estimators": [10, 100],
                 "criterion": ["gini", "entropy", "log_loss"],
             },
             ),
        step("mlp", MLPClassifier,
             space={
                 "activation": ["identity", "logistic", "relu"],
                 "alpha": (0.0001, 0.1),
                 "learning_rate": ["constant", "invscaling", "adaptive"],
             },
             ),
    ),
)


@dataclass
class FakedFittedAndValidatedBaseModel:
    name: str
    config: dict
    val_probabilities: List[np.ndarray]
    val_score: float
    le_: LabelEncoder
    classes_: np.ndarray

    def predict(self, X):
        # If we want to use such a fake base model at test time, we would need to return either predictions for test
        # or validation data using this function based e.g. on the X's hash. (It would be better to just use models.)
        return np.argmax(self.val_probabilities, axis=1)

    def predict_proba(self, X):
        return self.val_probabilities


def acc_loss(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return 1 - accuracy_score(y_true, y_pred)


def create_ensemble(bucket: PathBucket, /, n_iterations: int = 50, seed: int = 42) -> AbstractWeightedEnsemble:
    # -- Get validation data and build base models
    pattern = r"trial_(?P<trial>.+)_val_probabilities.npy"
    trial_names = [
        match.group("trial")
        for key in bucket
        if (match := re.match(pattern, key)) is not None
    ]

    X_train, X_val, X_test, y_train, y_val = (
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
    )

    # Store the classes seen during fit of base models (ask Lennart why, very specific edge case reason...)
    le_ = LabelEncoder().fit(y_train)
    classes_ = le_.classes_

    # Build "base models" to keep a fit/predict interface for the ensemble code
    #   Why? so that the interface for the ensemble is class- and model-based as one should use in practice.
    base_models = [
        FakedFittedAndValidatedBaseModel(
            name,
            bucket[f"trial_{name}_config.json"].load(),
            bucket[f"trial_{name}_val_probabilities.npy"].load(check=np.ndarray),
            bucket[f"trial_{name}_scores.json"].load()["validation_accuracy"],
            le_,
            classes_,

        )
        for name in trial_names
    ]

    # -- Pruning (Preprocessing for Post Hoc Ensembling)
    #  -> max_number_base_models=30 because we have 3 algorithms and thus each silo could have 10 at most
    base_models = prune_base_models(base_models, max_number_base_models=30, pruning_method="SiloTopN")

    print()
    # -- Greedy Ensemble Selection With Replacement
    # ges = GreedyEnsembleSelection(base_models, n_iterations=n_iterations, loss_function=acc_loss,
    #                               n_jobs=1, random_state=seed)
    # ges.fit(X_val, y_val)
    # ges.predict(X_test)  # Won't work because the faked base models don't return test predictions
    # return ges

    # -- Ensemble Weighting with CMA-ES
    ew_cma = EnsembleWeightingCMAES(base_models, n_iterations=n_iterations, loss_function=acc_loss, n_jobs=1,
                                    random_state=seed, normalize_weights="no", trim_weights="no",
                                    normalize_predict_proba_output=False)
    ew_cma.fit(X_val, y_val)
    # ew_cma.predict(X_test)  # Won't work because the faked base models don't return test predictions
    return ew_cma


def target_function(
        trial: Trial,
        /,
        bucket: PathBucket,
        pipeline: Pipeline,
) -> Trial.Report:
    X_train, X_val, X_test, y_train, y_val, y_test = (
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
        bucket["y_test.npy"].load(),
    )

    pipeline = pipeline.configure(trial.config)
    sklearn_pipeline = pipeline.build()

    # Begin the trial, the context block makes sure
    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

    if trial.exception:
        return trial.fail(cost=np.inf)

    # Make our predictions with the model
    train_predictions = sklearn_pipeline.predict(X_train)
    val_predictions = sklearn_pipeline.predict(X_val)
    test_predictions = sklearn_pipeline.predict(X_test)

    val_probabilites = sklearn_pipeline.predict_proba(X_val)

    # Save all of this to the file system
    scores = {
        "train_accuracy": accuracy_score(train_predictions, y_train),
        "validation_accuracy": accuracy_score(val_predictions, y_val),
        "test_accuracy": accuracy_score(test_predictions, y_test),
    }
    bucket.update(
        {
            f"trial_{trial.name}_config.json": dict(trial.config),
            f"trial_{trial.name}_scores.json": scores,
            f"trial_{trial.name}.pkl": sklearn_pipeline,
            f"trial_{trial.name}_val_predictions.npy": val_predictions,
            f"trial_{trial.name}_val_probabilities.npy": val_probabilites,
            f"trial_{trial.name}_test_predictions.npy": test_predictions,
        }
    )
    val_accuracy = scores["validation_accuracy"]
    return trial.success(cost=1 - val_accuracy)


def get_dataset() -> tuple[np.ndarray, ...]:
    dataset = openml.datasets.get_dataset(31)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.4
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, random_state=seed, test_size=0.5
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # type: ignore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed = 42

    path = Path("results")
    if path.exists():
        shutil.rmtree(path)

    # This bucket is just a convenient way to save and load data
    bucket = PathBucket(path)
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()

    # Save all of this to the file system
    bucket.update(
        {
            "X_train.csv": X_train,
            "X_val.csv": X_val,
            "X_test.csv": X_test,
            "y_train.npy": y_train,
            "y_val.npy": y_val,
            "y_test.npy": y_test,
        }
    )

    # For local
    n_workers = 4
    scheduler = Scheduler.with_processes(n_workers)

    # Set up the optimizer with our space
    optimizer = SMACOptimizer.HPO(space=pipeline.space(), seed=seed)

    # This objective is essentially a `partial` but well typed.
    objective = Trial.Objective(target_function, bucket=bucket, pipeline=pipeline)

    task = Trial.Task(objective, scheduler, concurrent_limit=n_workers)

    # We only want on of these running at a time
    ensemble_task = Task(create_ensemble, scheduler, concurrent_limit=1)

    # Below we can define a bunch of callbacks to define how our system
    # behaves. This could be done in some specialized class for your
    # use case but it's all spelled out here as we can't anticipate
    # every use case.
    reports: list[Trial.SuccessReport] = []
    ensembles: list[GreedyEnsembleSelection] = []


    @scheduler.on_start(repeat=n_workers)
    def launch_initial_tasks() -> None:
        """When we start, launch `n_workers` tasks."""
        trial = optimizer.ask()
        task(trial)


    @task.on_report
    def tell_optimizer(report: Trial.Report) -> None:
        """When we get a report, tell the optimizer."""
        optimizer.tell(report)


    @task.on_report
    def launch_another_task(_: Trial.Report) -> None:
        """When we get a report, evaluate another trial."""
        trial = optimizer.ask()
        task(trial)


    @task.on_report
    def log_report(report: Trial.Report) -> None:
        """When we get a report, log it."""
        logger.info(report)


    @task.on_success
    def save_success(report: Trial.SuccessReport) -> None:
        """When we get a report, save it."""
        reports.append(report)


    @task.on_success
    def launch_ensemble_task(_: Trial.SuccessReport) -> None:
        """When a task successfully completes, launch an ensemble task."""
        ensemble_task(bucket)


    @ensemble_task.on_returned
    def save_ensemble(ensemble) -> None:
        """When an ensemble task returns, save it."""
        ensembles.append(ensemble)


    @ensemble_task.on_exception
    def log_ensemble_exception(exception: BaseException) -> None:
        """When an ensemble task throws an exception, log it."""
        logger.exception(exception)


    @scheduler.on_timeout
    def run_last_ensemble_task() -> None:
        """When the scheduler is empty, run the last ensemble task."""
        ensemble_task(bucket)


    # We can specify a timeout of 60 seconds and tell it to wait
    # for all tasks to finish before stopping
    scheduler.run(timeout=60, wait=True)

    best_ensemble = min(ensembles, key=lambda e: e.validation_loss_)

    print("Best ensemble:")
    print(best_ensemble,
          best_ensemble.validation_loss_,
          best_ensemble.val_loss_over_iterations_)
