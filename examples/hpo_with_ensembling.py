"""Performing HPO with Post-Hoc Ensembling.
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn]`

This example performs Hyperparameter optimization on a fairly default
data-preprocessing + model sklearn pipeline, using a dataset pulled from
[OpenML](https://www.openml.org/d/31).

After the HPO is complete, we use the validation predictions from each trial
to create an ensemble using the
**Weighted Ensemble algorithm from Caruana et al. (2004)**.

??? quote "Reference: **Ensemble selection from libraries of models**"

    Rich Caruana, Alexandru Niculescu-Mizil, Geoff Crew and Alex Ksikes

    ICML 2004

    https://dl.acm.org/doi/10.1145/1015330.1015432

    https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf

This makes heavy use of the pipelines and the optimization faculties of
amltk. You can fine the [pipeline guide here](site:guides/pipelines.md)
and the [optimization guide here](site:guides/optimization.md) to learn more.

You can skip the imports sections and go straight to the
[pipeline definition](#pipeline-definition).

## Imports
"""
from __future__ import annotations

import shutil
from asyncio import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
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
from amltk.data.conversions import probabilities_to_classes

from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from amltk.optimization import History, Trial
from amltk.pipeline import Pipeline, choice, group, split, step
from amltk.scheduling import Scheduler, Task
from amltk.sklearn.data import split_data
from amltk.smac import SMACOptimizer
from amltk.store import PathBucket

"""
Below is just a small function to help us get the dataset from OpenML
and encode the labels.
"""


def get_dataset(seed: int) -> tuple[np.ndarray, ...]:
    dataset = openml.datasets.get_dataset(
        31,
        download_qualities=False,
        download_features_meta_data=False,
        download_data=True,
    )
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    _y = LabelEncoder().fit_transform(y)
    splits = split_data(  # <!> (1)!
        X,  # <!>
        _y,  # <!>
        splits={"train": 0.6, "val": 0.2, "test": 0.2},  # <!>
        seed=seed,  # <!>
    )  # <!>

    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]
    return x_train, x_val, x_test, y_train, y_val, y_test  # type: ignore


# 1. We use the `#!python split_data()` function from the
#   [sklearn integrations](site:reference/sklearn.md) to split the data into
#   a custom amount of splits, in this case
#   `#!python "train", "val", "test"`. You could also use the
#   dedicated [`train_val_test_split()`][amltk.sklearn.data.train_val_test_split]
#   function instead.
"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
a choice of classifier between a Random Forest, Support Vector Machine, and
Multi-Layer Perceptron.

For more on definitions of pipelines, see the [Pipeline](site:guides/pipeline.md)
guide.
"""
pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        group(  # <!> (3)!
            "categoricals",
            step(
                "category_imputer",
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
            ),
        ),
        group(  # <!> (2)!
            "numerics",
            step(
                "numerical_imputer",
                SimpleImputer,
                space={"strategy": ["mean", "median"]},
            )
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
            ),
        ),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    ),
    choice(  # <!> (1)!
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

print(pipeline)
print(pipeline.space())

# 1. Here we define a choice of algorithms to use where each entry is a possible
#   algorithm to use. Each algorithm is defined by a step, which is a
#   configuration of a sklearn estimator. The space parameter is a dictionary
#   of hyperparameters to optimize over, and the config parameter is a
#   dictionary of fixed parameters to set on the estimator.
# 2. Here we gropu the numerical preprocessing steps to use. Each step is a
#  scaler to use. Each scaler is defined by a step, which is a configuration
#  of the preprocessor. The space parameter is a dictionary of
#  hyperparameters to optimize over, and the config parameter is a dictionary
#  of fixed parameters to set on the preprocessing step.
# 3. Here we group the categorical preprocessing steps to use.
#   Each step is given a space, which is a dictionary of hyperparameters to
#   optimize over, and a config, which is a dictionary of fixed parameters to
#   set on the preprocessing step.
"""
## Target Function

Next we establish the actual target function we wish to evaluate, that is,
the function we wish to optimize. In this case, we are optimizing the
accuracy of the model on the validation set.

The target function takes a [`Trial`][amltk.optimization.Trial] object, which
has the configuration of the pipeline to evaluate and provides utility
to time, and return the results of the evaluation, whether it be a success
or failure.

We make use of a [`PathBucket`][amltk.store.PathBucket]
to store and load the data, and the `Pipeline` we defined above to
configure the pipeline with the hyperparameters we are optimizing over.

For more details, please check out the [Optimization](site:guides/optimization.md)
guide for more details.
"""


def target_function(
    trial: Trial,
    /,
    bucket: PathBucket,
    pipeline: Pipeline,
) -> Trial.Report:
    X_train, X_val, X_test, y_train, y_val, y_test = (  # (1)!
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
        bucket["y_test.npy"].load(),
    )
    pipeline = pipeline.configure(trial.config)  # <!> (2)!
    sklearn_pipeline = pipeline.build()  # <!>

    with trial.begin():  # <!> (3)!
        sklearn_pipeline.fit(X_train, y_train)

    if trial.exception:
        items = {
            "exception.txt": str(trial.exception),
            "config.json": dict(trial.config),
            "traceback.txt": str(trial.traceback),
        }

        trial.store(items, where=bucket)

        return trial.fail(cost=np.inf)  # <!> (4)!

    # Make our predictions with the model
    train_predictions = sklearn_pipeline.predict(X_train)
    val_predictions = sklearn_pipeline.predict(X_val)
    test_predictions = sklearn_pipeline.predict(X_test)

    val_probabilites = sklearn_pipeline.predict_proba(X_val)
    val_accuracy = accuracy_score(val_predictions, y_val)

    # Save the scores to the summary of the trial
    trial.summary.update(
        {
            "train_accuracy": accuracy_score(train_predictions, y_train),
            "validation_accuracy": val_accuracy,
            "test_accuracy": accuracy_score(test_predictions, y_test),
        },
    )

    # Save all of this to the file system
    trial.store(  # (5)!
        {
            "config.json": dict(trial.config),
            "scores.json": trial.summary,
            "model.pkl": sklearn_pipeline,
            "val_predictions.npy": val_predictions,
            "val_probabilities.npy": val_probabilites,
            "test_predictions.npy": test_predictions,
        },
        where=bucket,
    )

    return trial.success(cost=1 - val_accuracy)  # <!> (6)!


# 1. We can easily load data from a [`PathBucket`][amltk.store.PathBucket]
#   using the `load` method.
# 2. We configure the pipeline with a specific set of hyperparameters suggested
#  by the optimizer through the [`Trial`][amltk.optimization.Trial] object.
# 3. We begin the trial by timing the execution of the target function and capturing
#  any potential exceptions.
# 4. If the trial failed, we return a failed report with a cost of infinity.
# 5. We save the results of the trial using
#   [`Trial.store`][amltk.optimization.Trial.store], creating a subdirectory
#   for this trial.
# 6. We return a successful report with the cost of the trial, which is the
# inverse of the validation accuracy.
"""
Next we define a simple [`@dataclass`][dataclasses.dataclass] to store the
our definition of an Esemble, which is simply a collection of the models trial
names to their weight in the ensemble. We also store the trajectory of the
ensemble, which is a list of tuples of the trial name and the weight of the
trial at that point in the trajectory. Finally, we store the configuration
of each trial in the ensemble.

We could of course add extra functionality to the Ensemble, give it references
to the [`PathBucket`][amltk.store.PathBucket] and [`Pipeline`][amltk.pipeline.Pipeline]
objects, and even add methods to train the ensemble, but for the sake of
simplicity we will leave it as is.
"""


@dataclass
class Ensemble:
    weights: dict[str, float]
    trajectory: list[tuple[str, float]]
    configs: dict[str, dict[str, Any]]


def create_ensemble(
    history: History,
    bucket: PathBucket,
    /,
    size: int = 5,
    seed: int = 42,
) -> Ensemble:
    if len(history) == 0:
        return Ensemble({}, [], {})

    validation_predictions = {
        name: report.retrieve("val_probabilities.npy", where=bucket)
        for name, report in history.items()
    }
    targets = bucket["y_val.npy"].load()

    accuracy: Callable[[np.ndarray, np.ndarray], float] = accuracy_score  # type: ignore

    def _score(_targets: np.ndarray, ensembled_probabilities: np.ndarray) -> float:
        predictions = probabilities_to_classes(ensembled_probabilities, classes=[0, 1])
        return accuracy(_targets, predictions)

    weights, trajectory, final_probabilities = weighted_ensemble_caruana(  # <!>
        model_predictions=validation_predictions,  # <!>
        targets=targets,  # <!>
        size=size,  # <!>
        metric=_score,  # <!>
        select=max,  # <!>
        seed=seed,  # <!>
    )  # <!>

    configs = {
        name: history[name].retrieve("config.json", where=bucket) for name in weights
    }
    return Ensemble(weights=weights, trajectory=trajectory, configs=configs)


"""
## Main
Finally we come to the main script that runs everything.
"""
seed = 42

X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(seed)  # (1)!

path = Path("hpo_with_ensembling_results")
if path.exists():
    shutil.rmtree(path)

bucket = PathBucket(path)
bucket.store(  # (2)!
    {
        "X_train.csv": X_train,
        "X_val.csv": X_val,
        "X_test.csv": X_test,
        "y_train.npy": y_train,
        "y_val.npy": y_val,
        "y_test.npy": y_test,
    },
)

scheduler = Scheduler.with_processes()  # (3)!
optimizer = SMACOptimizer.create(space=pipeline.space(), seed=seed)  # (4)!

task = Task(target_function, scheduler)  # (6)!
ensemble_task = Task(create_ensemble, scheduler)  # (7)!

trial_history = History()
ensembles: list[Ensemble] = []


@scheduler.on_start  # (8)!
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task.submit(trial, bucket=bucket, pipeline=pipeline)


@task.on_returned
def tell_optimizer(future: Future, report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)


@task.on_returned
def add_to_history(future: Future, report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)


@task.on_returned
def launch_ensemble_task(future: Future, report: Trial.Report) -> None:
    """When a task successfully completes, launch an ensemble task."""
    if report.status is Trial.Status.SUCCESS:
        ensemble_task(trial_history, bucket)


@task.on_returned
def launch_another_task(*_: Any) -> None:
    """When we get a report, evaluate another trial."""
    trial = optimizer.ask()
    task(trial, bucket=bucket, pipeline=pipeline)


@ensemble_task.on_returned
def save_ensemble(future: Future, ensemble: Ensemble) -> None:
    """When an ensemble task returns, save it."""
    ensembles.append(ensemble)


@task.on_exception
@ensemble_task.on_exception
def print_ensemble_exception(future: Future[Any], exception: BaseException) -> None:
    """When an exception occurs, log it and stop."""
    print(exception)
    scheduler.stop()


@scheduler.on_timeout
def run_last_ensemble_task() -> None:
    """When the scheduler is empty, run the last ensemble task."""
    ensemble_task(trial_history, bucket)


scheduler.run(timeout=5, wait=True)  # (9)!

print("Trial history:")
history_df = trial_history.df()
print(history_df)

best_ensemble = max(ensembles, key=lambda e: e.trajectory[-1])

print("Best ensemble:")
print(best_ensemble)
# 1. We use `#!python get_dataset()` defined earlier to load the
#  dataset.
# 2. We use [`store()`][amltk.store.Bucket.store] to store the data in the bucket, with
# each key being the name of the file and the value being the data.
# 3. We use [`Scheduler.with_processes()`][amltk.scheduling.Scheduler.with_processes]
#  create a [`Scheduler`][amltk.scheduling.Scheduler] that runs everything
#  in a different process. You can of course use a different backend if you want.
# 4. We use [`SMACOptimizer.create()`][amltk.smac.SMACOptimizer.create] to create a
#  [`SMACOptimizer`][amltk.smac.SMACOptimizer] given the space from the pipeline
#  to optimize over.
# 6. We create a [`Task`][amltk.scheduling.Task] that will run our objective, passing
#   in the function to run and the scheduler for where to run it
# 7. We use [`Task()`][amltk.scheduling.Task] to create a
#   [`Task`][amltk.scheduling.Task]
#   for the `create_ensemble` method above. This will also run in parallel with the hpo
#   trials if using a non-sequential scheduling mode.
# 8. We use `Scheduler.on_start()` hook to register a
#  callback that will be called when the scheduler starts. We can use the
#  `repeat` argument to make sure it's called many times if we want.
# 9. We use [`Scheduler.run()`][amltk.scheduling.Scheduler.run] to run the scheduler.
#  Here we set it to run briefly for 5 seconds and wait for remaining tasks to finish
#  before continuing.
