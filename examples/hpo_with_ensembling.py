from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from ConfigSpace import Configuration
import numpy as np
import openml
from dask_jobqueue import SLURMCluster
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
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from byop.control import AskAndTell
from byop.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from byop.optimization import RandomSearch
from byop.pipeline import Pipeline, choice, split, step
from byop.scheduling import Scheduler
from byop.spaces import ConfigSpaceSampler
from byop.store import PathBucket

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
        step(
            "tabpfn",
            TabPFNClassifier,
            space={"N_ensemble_configurations": [1,2,3,4] },
            config = {"device": "cpu", "seed":42},
        )
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
    trials = [
        match.group("trial")
        for key in bucket
        if (match := re.match(r"trial_(?P<trial>\d+)", key)) is not None
    ]
    validation_predictions = {
        trial: bucket[f"trial_{trial}_val_probabilities.npy"].load() for trial in trials
    }
    targets = bucket["y_val.npy"].load()
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score  # type: ignore

    weights, trajectory = weighted_ensemble_caruana(
        model_predictions=validation_predictions,
        targets=targets,
        size=size,
        metric=metric,
        select=max,
        seed=seed,
        is_probabilities=True,
        classes=[0, 1],
    )

    configs = {key: bucket[f"trial_{key}_config.json"].load() for key in weights}
    ensemble = Ensemble(weights, trajectory, configs)
    logger.debug(f"{ensemble=}")
    return ensemble


def target_function(
    trial_number: int,
    config: Configuration,
    /,
    bucket: PathBucket,
    pipeline: Pipeline,
) -> float:
    X_train, X_val, X_test, y_train, y_val, y_test = (
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
        bucket["y_test.npy"].load(),
    )

    pipeline = pipeline.configure(config)
    sklearn_pipeline = pipeline.build()

    sklearn_pipeline.fit(X_train, y_train)

    train_predictions = sklearn_pipeline.predict(X_train)
    val_predictions = sklearn_pipeline.predict(X_val)
    val_probabilites = sklearn_pipeline.predict_proba(X_val)
    test_predictions = sklearn_pipeline.predict(X_test)

    train_accuracy = accuracy_score(train_predictions, y_train)
    val_accuracy = accuracy_score(val_predictions, y_val)
    test_accuracy = accuracy_score(test_predictions, y_test)

    trial_name = f"trial_{trial_number}"
    # Save all of this to the file system
    scores = {
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
    }
    bucket.update(
        {
            f"{trial_name}_config.json": dict(config),
            f"{trial_name}_scores.json": scores,
            f"{trial_name}.pkl": sklearn_pipeline,
            f"{trial_name}_val_predictions.npy": val_predictions,
            f"{trial_name}_val_probabilities.npy": val_probabilites,
            f"{trial_name}_test_predictions.npy": test_predictions,
        }
    )
    assert isinstance(val_accuracy, float)
    return val_accuracy


if __name__ == "__main__":
    seed = 42

    space = pipeline.space(seed=seed, parser="configspace")

    path = Path("results")
    if path.exists():
        shutil.rmtree(path)

    bucket = PathBucket(path)

    dataset = openml.datasets.get_dataset(31)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    print(X, y)
    print(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.4
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, random_state=seed, test_size=0.5
    )
    le = LabelEncoder().fit(y)

    # Save all of this to the file system
    bucket.update(
        {
            "X_train.csv": X_train,
            "X_val.csv": X_val,
            "X_test.csv": X_test,
            "y_train.npy": le.transform(y_train),
            "y_val.npy": le.transform(y_val),
            "y_test.npy": le.transform(y_test),
        }
    )

    here = Path(__file__).absolute().parent
    logs = here / "logs-test-dask-slurm"
    logs.mkdir(exist_ok=True)
    
    # For testing out slurm
    #n_workers = 256
    #SLURMCluster.job_cls.submit_command = "sbatch --bosch"
    #cluster = SLURMCluster(
    #    memory="2GB",
    #    processes=1,
    #    cores=1,
    #    local_directory=here,
    #    log_directory=logs,
    #    queue="bosch_cpu-cascadelake",
    #    job_extra_directives=["--time 0-00:10:00"]
    #)
    #cluster.adapt(maximum_jobs=n_workers)
    #executor = cluster.get_client().get_executor()
    #scheduler = Scheduler(executor=executor)

    # For local
    n_workers = 4
    scheduler = Scheduler.with_processes(n_workers)
    rs = RandomSearch(space=space, sampler=ConfigSpaceSampler)
    objective = AskAndTell.objective(target_function, bucket=bucket, pipeline=pipeline)

    controller = AskAndTell(
        objective=objective,
        scheduler=scheduler,
        optimizer=rs,
        max_trials=n_workers * 2,
        concurrent_trials=n_workers - 1,
    )

    val_results: list[float] = []
    ensembles: list[Ensemble] = []

    controller.trial.on_success(val_results.append)
    controller.trial.on_error(logger.error)

    # Your ensembling method HERE
    ensemble_task = scheduler.task(create_ensemble, name="ensembling")
    ensemble_task.on_success(ensembles.append)

    def maybe_launch_ensemble_task(_):
        if not ensemble_task.running():
            ensemble_task(bucket, size=10, seed=seed)

    controller.trial.on_success(maybe_launch_ensemble_task)
    scheduler.on_empty(lambda: ensemble_task(bucket, size=10, seed=seed))
    controller.run()

    print(scheduler.counts)
    print(val_results)
    print([e.trajectory[-1] for e in ensembles])
