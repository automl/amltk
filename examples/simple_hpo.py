"""Simple HPO loop
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn]`

This example shows the basic of setting up a simple HPO loop around a
`RandomForestClassifier`. We will use the [OpenML](https://openml.org) to
get a dataset and also use some static preprocessing as part of our pipeline
definition.

You can fine the [pipeline guide here](../../guides/pipelines)
and the [optimization guide here](../../guides/optimization) to learn more.

You can skip the imports sections and go straight to the
[pipeline definition](#pipeline-definition).

## Imports
"""
from __future__ import annotations

from typing import Any

import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
)

from amltk.optimization import History, Trial
from amltk.pipeline import Pipeline, split, step
from amltk.scheduling import Scheduler
from amltk.sklearn.data import split_data
from amltk.smac import SMACOptimizer
from amltk.store import PathBucket

"""
## Dataset

Below is just a small function to help us get the dataset from OpenML and encode the
labels.
"""


def get_dataset(
    dataset_id: str | int,
    *,
    seed: int,
    splits: dict[str, float],
) -> dict[str, Any]:
    dataset = openml.datasets.get_dataset(dataset_id)

    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)

    return split_data(X, _y, splits=splits, seed=seed)  # type: ignore


"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
the `RandomForestClassifier`.

For more on definitions of pipelines, see the [Pipeline](../../guides/pipeline)
guide.
"""
categorical_imputer = step(
    "categoricals",
    SimpleImputer,
    config={
        "strategy": "constant",
        "fill_value": "missing",
    },
)
one_hot_encoding = step("ohe", OneHotEncoder, config={"drop": "first"})

numerical_imputer = step(
    "numerics",
    SimpleImputer,
    space={"strategy": ["mean", "median"]},
)

pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        categorical_imputer | one_hot_encoding,
        numerical_imputer,
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    ),
    step(
        "rf",
        RandomForestClassifier,
        space={
            "n_estimators": (10, 100),
            "criterion": ["gini", "entropy", "log_loss"],
        },
    ),
)

print(pipeline)
print(pipeline.space())

"""
## Target Function
The function we will optimize must take in a `Trial` and return a `Trial.Report`.
We also pass in a [`PathBucket`][amltk.store.Bucket] which is a dict-like view of the
file system, where we have our dataset stored.

We also pass in our [`Pipeline`][amltk.pipeline.Pipeline] representation of our
pipeline, which we will use to build our sklearn pipeline with a specific
`trial.config` suggested by the [`Optimizer`][amltk.optimization.Optimizer].
"""


def target_function(
    trial: Trial,
    /,
    bucket: PathBucket,
    _pipeline: Pipeline,
) -> Trial.Report:
    # Load in data
    X_train, X_val, X_test, y_train, y_val, y_test = (
        bucket["X_train.csv"].load(),
        bucket["X_val.csv"].load(),
        bucket["X_test.csv"].load(),
        bucket["y_train.npy"].load(),
        bucket["y_val.npy"].load(),
        bucket["y_test.npy"].load(),
    )

    # Configure the pipeline with the trial config before building it.
    configured_pipeline = _pipeline.configure(trial.config)
    sklearn_pipeline = configured_pipeline.build()

    # Fit the pipeline, indicating when you want to start the trial timing and error
    # catchnig.
    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

    # If an exception happened, we use `trial.fail` to indicate that the
    # trial failed
    if trial.exception:
        trial.store(
            {
                "exception.txt": f"{trial.exception}\n traceback: {trial.traceback}",
                "config.json": dict(trial.config),
            },
            where=bucket,
        )
        return trial.fail(cost=np.inf)

    # Make our predictions with the model
    train_predictions = sklearn_pipeline.predict(X_train)
    val_predictions = sklearn_pipeline.predict(X_val)
    test_predictions = sklearn_pipeline.predict(X_test)

    val_probabilites = sklearn_pipeline.predict_proba(X_val)

    # Save the scores to the summary of the trial
    val_accuracy = accuracy_score(val_predictions, y_val)
    trial.summary.update(
        {
            "train/acc": accuracy_score(train_predictions, y_train),
            "val/acc": val_accuracy,
            "test/acc": accuracy_score(test_predictions, y_test),
        },
    )

    # Save all of this to the file system
    trial.store(
        {
            "config.json": dict(trial.config),
            "scores.json": trial.summary,
            "model.pkl": sklearn_pipeline,
            "val_probabilities.npy": val_probabilites,
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
        where=bucket,
    )

    # Finally report the success
    return trial.success(cost=1 - val_accuracy)


"""
## Running the Whole Thing

Now we can run the whole thing. We will use the
[`Scheduler`][amltk.scheduling.Scheduler]
to run the optimization, and the [`SMACOptimizer`][amltk.smac.SMACOptimizer] to
to optimize the pipeline.

### Getting and storing data
We use a [`PathBucket`][amltk.store.PathBucket] to store the data. This is a dict-like
view of the file system.
"""

seed = 42
data = get_dataset(31, seed=seed, splits={"train": 0.6, "val": 0.2, "test": 0.2})

X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]

bucket = PathBucket("results/simple_hpo_example", clean=True, create=True)
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

print(bucket)
print(dict(bucket))

X_train = bucket["X_train.csv"].load()
print(X_train.shape)

"""
### Setting up the Scheduler, Task and Optimizer
We use the [`Scheduler.with_sequential`][amltk.scheduling.Scheduler.with_sequential]
method to create a [`Scheduler`][amltk.scheduling.Scheduler] that will run the
optimization sequentially and in the same process. This is useful for debugging.

Please check out the full [guides](../../guides) to learn more!

We then create an [`SMACOptimizer`][amltk.smac.SMACOptimizer] which will
optimize the pipeline. We pass in the space of the pipeline, which is the space of
the hyperparameters we want to optimize.
"""
scheduler = Scheduler.with_sequential()
optimizer = SMACOptimizer.create(space=pipeline.space(), seed=seed)


"""
Next we create a [`Trial.Task`][amltk.optimization.Trial.Task] which is a special kind
of [`Task`][amltk.scheduling.Task] with events that are useful for optimization loops.
We pass it in the function we want to run and the scheduler we will run it in.
"""
task = Trial.task(target_function, scheduler)

print(task)
"""
We use the callback decorators of the [`Scheduler`][amltk.scheduling.Scheduler] and
the [`Trial.Task`][amltk.optimization.Trial.Task] to add callbacks that get called
during events that happen during the running of the scheduler. Using this, we can
control the flow of how things run.
Check out the [task guide](../../guides/tasks) for more.

This one here asks the optimizer for a new trial when the scheduler starts and
launches the task we created earlier with this trial.
"""


@scheduler.on_start
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task(trial, bucket=bucket, _pipeline=pipeline)


"""
When a [`Trial.Task`][amltk.optimization.Trial.Task] returns and we get a report, i.e.
with [`task.success()`][amltk.optimization.Trial.success] or
[`task.fail()`][amltk.optimization.Trial.fail], the `task` will fire off the
callbacks registered with [`.on_success()`][amltk.optimization.Trial.Task.on_success] or
[`.on_failed()`][amltk.optimization.Trial.Task.on_failed]
respectively, with a general [`.on_report()`][amltk.optimization.Trial.Task.on_report]
callback for both. We can use these to add callbacks that get called when these
events happen.

Here we use it to update the optimizer with the report we got.
"""


@task.on_report
def tell_optimizer(report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)


"""
We can use the [`History`][amltk.optimization.History] class to store the reports we get
from the [`Trial.Task`][amltk.optimization.Trial.Task]. We can then use this to analyze
the results of the optimization afterwords.
"""
trial_history = History()


@task.on_report
def add_to_history(report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)


"""
We launch a new task when the scheduler is empty, i.e. when all the tasks have
finished. This will keep going until we hit the timeout we set on the scheduler.

If you want to run the optimization in parallel, you can use the
`task.on_report` callback to launch a new task when you get a report. This will
launch a new task as soon as one finishes.
"""


@scheduler.on_empty
def launch_another_task() -> None:
    """When we get a report, evaluate another trial."""
    trial = optimizer.ask()
    task(trial, bucket=bucket, _pipeline=pipeline)


"""
### Setting the system to run

Lastly we use [`Scheduler.run`][amltk.scheduling.Scheduler.run] to run the
scheduler. We pass in a timeout of 5 seconds.
"""
scheduler.run(timeout=20)

print("Trial history:")
history_df = trial_history.df()
print(history_df)
