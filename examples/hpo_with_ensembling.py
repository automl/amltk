from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# from dask_jobqueue import SLURMCluster
from byop.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
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

pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        (
            step(
                "categoricals",
                SimpleImputer,
                space={
                    "strategy": ["most_frequent", "constant"],
                    "fill_value": ["missing"],
                },
            )
            | step(
                "ohe",
                OneHotEncoder,
                space={
                    "min_frequency": (0.01, 0.1),
                    "handle_unknown": ["ignore", "infrequent_if_exist"],
                },
                config={"drop": "first"},
            )
        ),
        (
            step("numerics", SimpleImputer, space={"strategy": ["mean", "median"]})
            | step(
                "variance_threshold",
                VarianceThreshold,
                space={"threshold": (0.0, 0.2)},
            )
            | choice(
                "scaler",
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
    choice(
        "algorithm",
        step("svm", SVC, space={"C": (0.1, 10.0)}, config={"probability": True}),
        step(
            "rf",
            RandomForestClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),
        step(
            "mlp",
            MLPClassifier,
            space={
                "activation": ["identity", "logistic", "relu"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
        ),
    ),
)


@dataclass
class Ensemble:
    weights: dict[str, float]
    trajectory: list[tuple[str, float]]
    configs: dict[str, dict[str, Any]]


def create_ensemble(
    bucket: PathBucket,
    /,
    size: int = 5,
    seed: int = 42,
) -> Ensemble:
    pattern = r"trial_(?P<trial>.+)_val_probabilities.npy"
    trial_names = [
        match.group("trial")
        for key in bucket
        if (match := re.match(pattern, key)) is not None
    ]
    validation_predictions = {
        name: bucket[f"trial_{name}_val_probabilities.npy"].load(check=np.ndarray)
        for name in trial_names
    }
    targets = bucket["y_val.npy"].load()

    task = Trial.Task
    trajectory: list[tuple[str, float]]
    weights, trajectory = weighted_ensemble_caruana(
        model_predictions=validation_predictions,
        targets=targets,
        size=size,
        metric=accuracy_score,
        select=max,  # type: ignore
        seed=seed,
        is_probabilities=True,
        classes=[0, 1],
    )

    configs = {key: bucket[f"trial_{key}_config.json"].load() for key in weights}
    ensemble = Ensemble(weights, trajectory, configs)
    logger.debug(f"{ensemble=}")
    return ensemble


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

    # For testing out slurm
    # n_workers = 256
    # here = Path(__file__).absolute().parent
    # logs = here / "logs-test-dask-slurm"
    # logs.mkdir(exist_ok=True)
    #
    # SLURMCluster.job_cls.submit_command = "sbatch --bosch"
    # cluster = SLURMCluster(
    #    memory="2GB",
    #    processes=1,
    #    cores=1,
    #    local_directory=here,
    #    log_directory=logs,
    #    queue="bosch_cpu-cascadelake",
    #    job_extra_directives=["--time 0-00:10:00"]
    # )
    # cluster.adapt(maximum_jobs=n_workers)
    # executor = cluster.get_client().get_executor()
    # scheduler = Scheduler(executor=executor)

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
    ensembles: list[Ensemble] = []

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
    def save_ensemble(ensemble: Ensemble) -> None:
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
    scheduler.run(timeout=5, wait=True)

    best_ensemble = max(ensembles, key=lambda e: e.trajectory[-1])

    print("Best ensemble:")
    print(best_ensemble)
