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
byop. You can fine the [pipeline guide here](../../guides/pipelines)
and the [optimization guide here](../../guides/optimization) to learn more.

You can skip the imports sections and go straight to the
[pipeline definition](#pipeline-definition).

## Imports
"""
from __future__ import annotations

import shutil
import traceback
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

from byop.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from byop.optimization import History, Trial
from byop.pipeline import Pipeline, choice, split, step
from byop.scheduling import Scheduler, Task
from byop.sklearn.data import split_data
from byop.smac import SMACOptimizer
from byop.store import PathBucket

"""
Below is just a small function to help us get the dataset from OpenML
and encode the labels.
"""


def get_dataset(seed: int) -> tuple[np.ndarray, ...]:
    dataset = openml.datasets.get_dataset(31)
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
#   [sklearn integrations](../../sklearn) to split the data into
#   a custom amount of splits, in this case
#   `#!python "train", "val", "test"`. You could also use the
#   dedicated [`train_val_test_split()`][byop.sklearn.data.train_val_test_split]
#   function instead.
"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
a choice of classifier between a Random Forest, Support Vector Machine, and
Multi-Layer Perceptron.

For more on definitions of pipelines, see the [Pipeline](../../guides/pipeline)
guide.
"""
pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        (  # <!> (3)!
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
        (  # <!> (2)!
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
# 2. Here we define the numerical preprocessing steps to use. Each step is a
#  scaler to use. Each scaler is defined by a step, which is a configuration
#  of the preprocessor. The space parameter is a dictionary of
#  hyperparameters to optimize over, and the config parameter is a dictionary
#  of fixed parameters to set on the preprocessing step.
# 3. Here we define the categorical preprocessing steps to use.
#   Each step is given a space, which is a dictionary of hyperparameters to
#   optimize over, and a config, which is a dictionary of fixed parameters to
#   set on the preprocessing step.
"""
## Target Function

Next we establish the actual target function we wish to evaluate, that is,
the function we wish to optimize. In this case, we are optimizing the
accuracy of the model on the validation set.

The target function takes a [`Trial`][byop.optimization.Trial] object, which
has the configuration of the pipeline to evaluate and provides utility
to time, and return the results of the evaluation, whether it be a success
or failure.

We make use of a [`PathBucket`][byop.store.PathBucket]
to store and load the data, and the `Pipeline` we defined above to
configure the pipeline with the hyperparameters we are optimizing over.

For more details, please check out the [Optimization](../../guides/optimization)
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
        trial.store(
            {
                "exception.txt": str(trial.exception),
                "traceback.txt": traceback.format_tb(trial.traceback),
                "config.json": dict(trial.config),
            },
            where=bucket,
        )
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


# 1. We can easily load data from a [`PathBucket`][byop.store.PathBucket]
#   using the `load` method.
# 2. We configure the pipeline with a specific set of hyperparameters suggested
#  by the optimizer through the [`Trial`][byop.optimization.Trial] object.
# 3. We begin the trial by timing the execution of the target function and capturing
#  any potential exceptions.
# 4. If the trial failed, we return a failed report with a cost of infinity.
# 5. We save the results of the trial using
#   [`Trial.store`][byop.optimization.Trial.store], creating a subdirectory
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
to the [`PathBucket`][byop.store.PathBucket] and [`Pipeline`][byop.pipeline.Pipeline]
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

    weights, trajectory = weighted_ensemble_caruana(  # <!>
        model_predictions=validation_predictions,  # <!>
        targets=targets,  # <!>
        size=size,  # <!>
        metric=accuracy,  # <!>
        select=max,  # <!>
        seed=seed,  # <!>
        is_probabilities=True,  # <!>
        classes=[0, 1],  # <!>
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

scheduler = Scheduler.with_sequential()  # (3)!
optimizer = SMACOptimizer.HPO(space=pipeline.space(), seed=seed)  # (4)!

objective = Trial.Objective(  # (5)!
    target_function,
    bucket=bucket,
    pipeline=pipeline,
)
task = Trial.Task(objective, scheduler)  # (6)!

ensemble_task = Task(create_ensemble, scheduler)  # (7)!

trial_history = History()
ensembles: list[Ensemble] = []


@scheduler.on_start  # (8)!
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task(trial)


@task.on_report
def tell_optimizer(report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)

@task.on_report
def add_to_history(report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)

@task.on_success
def launch_ensemble_task(_: Trial.SuccessReport) -> None:
    """When a task successfully completes, launch an ensemble task."""
    ensemble_task(trial_history, bucket)


@task.on_report
def launch_another_task(_: Trial.Report) -> None:
    """When we get a report, evaluate another trial."""
    trial = optimizer.ask()
    task(trial)



@ensemble_task.on_returned
def save_ensemble(ensemble: Ensemble) -> None:
    """When an ensemble task returns, save it."""
    ensembles.append(ensemble)


@ensemble_task.on_exception
def print_ensemble_exception(exception: BaseException) -> None:
    """When an ensemble task throws an exception, log it."""
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
# 2. We use [`store()`][byop.store.Bucket.store] to store the data in the bucket, with
# each key being the name of the file and the value being the data.
# 3. We use [`Scheduler.with_sequential()`][byop.scheduling.Scheduler.with_sequential]
#  create a [`Scheduler`][byop.scheduling.Scheduler] that runs everything
#  sequentially. You can of course use a different backend if you want.
# 4. We use [`SMACOptimizer.HPO()`][byop.smac.SMACOptimizer.HPO] to create a
#  [`SMACOptimizer`][byop.smac.SMACOptimizer] given the space from the pipeline
#  to optimize over.
# 5. We use [`Trial.Objective()`][byop.optimization.Trial.Objective] to wrap our
# `target_function` in a [`Trial.Task`][byop.optimization.Trial.Task].
#
#     !!! note "`Trial.Objective`"
#
#         While a [partial][functools.partial] also works, using
#         [`Trial.Objective`][byop.optimization.Trial.Objective] let's us
#         maintain some type safety with mypy.
#
# 6. We use [`Trial.Task()`][byop.optimization.Trial.Task] to on our objective
#   to create a [`Task`][byop.scheduling.Task] that will run our objective.
# 7. We use [`Task()`][byop.scheduling.Task] to create a [`Task`][byop.scheduling.Task]
#   for the `create_ensemble` method above. This will also run in parallel with the hpo
#   trials if using a non-sequential scheduling mode.
# 8. We use `Scheduler.on_start()` hook to register a
#  callback that will be called when the scheduler starts. We can use the
#  `repeat` argument to make sure it's called many times if we want.
# 9. We use [`Scheduler.run()`][byop.scheduling.Scheduler.run] to run the scheduler.
#  Here we set it to run briefly for 5 seconds and wait for remaining tasks to finish
#  before continuing.
