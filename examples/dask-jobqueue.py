"""Using the Scheduler with SLURM (dask-jobqueue)

The point of this example is to show
how to set up `dask-jobqueue` with a realistic workload.

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn, dask-jobqueue]`

This example shows how to use `dask-jobqueue` to run HPO on a
`RandomForestClassifier` with SMAC. This workload is borrowed from
the HPO example.

SMAC can not handle fast updates and seems to be quite
efficient for this workload with ~32 cores.
"""

from typing import Any

import openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Component, Node, Sequential, Split
from amltk.scheduling import Scheduler
from amltk.sklearn import split_data
from amltk.store import PathBucket

N_WORKERS = 32
scheduler = Scheduler.with_slurm(
    n_workers=N_WORKERS,  # Number of workers to launch
    queue="the-name-of-the-partition/queue",  # Name of the queue to submit to
    cores=1,  # Number of cores per worker
    memory="4 GB",  # Memory per worker
    walltime="00:20:00",  # Walltime per worker
    # submit_command="sbatch --extra-arguments",  # Sometimes you need extra arguments to the launch command
)


def get_dataset(
    dataset_id: str | int,
    *,
    seed: int,
    splits: dict[str, float],
) -> dict[str, Any]:
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_features_meta_data=False,
        download_qualities=False,
    )
    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)
    return split_data(X, _y, splits=splits, seed=seed)  # type: ignore


pipeline = (
    Sequential(name="Pipeline")
    >> Split(
        {
            "categorical": [
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(drop="first"),
            ],
            "numerical": Component(
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            ),
        },
        name="feature_preprocessing",
    )
    >> Component(
        RandomForestClassifier,
        space={
            "n_estimators": (10, 100),
            "max_features": (0.0, 1.0),
            "criterion": ["gini", "entropy", "log_loss"],
        },
    )
)


def target_function(trial: Trial, _pipeline: Node) -> Trial.Report:
    trial.store({"config.json": trial.config})
    with trial.profile("data-loading"):
        X_train, X_val, X_test, y_train, y_val, y_test = (
            trial.bucket["X_train.csv"].load(),
            trial.bucket["X_val.csv"].load(),
            trial.bucket["X_test.csv"].load(),
            trial.bucket["y_train.npy"].load(),
            trial.bucket["y_val.npy"].load(),
            trial.bucket["y_test.npy"].load(),
        )

    sklearn_pipeline = _pipeline.configure(trial.config).build("sklearn")

    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

    if trial.exception:
        trial.store({"exception.txt": f"{trial.exception}\n {trial.traceback}"})
        return trial.fail()

    with trial.profile("predictions"):
        train_predictions = sklearn_pipeline.predict(X_train)
        val_predictions = sklearn_pipeline.predict(X_val)
        test_predictions = sklearn_pipeline.predict(X_test)

    with trial.profile("probabilities"):
        val_probabilites = sklearn_pipeline.predict_proba(X_val)

    with trial.profile("scoring"):
        train_acc = float(accuracy_score(train_predictions, y_train))
        val_acc = float(accuracy_score(val_predictions, y_val))
        test_acc = float(accuracy_score(test_predictions, y_test))

    trial.summary["train/acc"] = train_acc
    trial.summary["val/acc"] = val_acc
    trial.summary["test/acc"] = test_acc

    trial.store(
        {
            "model.pkl": sklearn_pipeline,
            "val_probabilities.npy": val_probabilites,
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
    )

    return trial.success(accuracy=val_acc)


seed = 42
data = get_dataset(31, seed=seed, splits={"train": 0.6, "val": 0.2, "test": 0.2})

X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]

bucket = PathBucket("example-hpo", clean=True, create=True)
bucket.store(
    {
        "X_train.csv": X_train,
        "X_val.csv": X_val,
        "X_test.csv": X_test,
        "y_train.npy": y_train,
        "y_val.npy": y_val,
        "y_test.npy": y_test,
    },
)


optimizer = SMACOptimizer.create(
    space=pipeline,  # <!> (1)!
    metrics=Metric("accuracy", minimize=False, bounds=(0.0, 1.0)),
    bucket=bucket,
    seed=seed,
)
task = scheduler.task(target_function)


@scheduler.on_start(repeat=N_WORKERS)
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task.submit(trial, _pipeline=pipeline)


trial_history = History()


@task.on_result
def process_result_and_launc(_, report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)
    optimizer.tell(report)
    if scheduler.running():
        trial = optimizer.ask()
        task.submit(trial, _pipeline=pipeline)


@task.on_cancelled
def stop_scheduler_on_cancelled(_: Any) -> None:
    raise RuntimeError("Scheduler cancelled a worker!")


if __name__ == "__main__":
    scheduler.run(timeout=60)

    history_df = trial_history.df()
    print(history_df)
    print(len(history_df))
