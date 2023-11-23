"""HPO
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn]`

This example shows the basic of setting up a simple HPO loop around a
`RandomForestClassifier`. We will use the [OpenML](https://openml.org) to
get a dataset and also use some static preprocessing as part of our pipeline
definition.

You can fine the [pipeline guide here](../guides/pipelines.md)
and the [optimization guide here](../guides/optimization.md) to learn more.

You can skip the imports sections and go straight to the
[pipeline definition](#pipeline-definition).

## Dataset

Below is just a small function to help us get the dataset from OpenML and encode the
labels.
"""

from typing import Any

import openml
from sklearn.preprocessing import LabelEncoder

from amltk.sklearn import split_data


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


"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
the `RandomForestClassifier`.

For more on definitions of pipelines, see the [Pipeline](../guides/pipelines.md)
guide.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from amltk.pipeline import Component, Node, Sequential, Split

pipeline = (
    Sequential(name="Pipeline")
    >> Split(
        {
            "categorical": [SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(drop="first")],
            "numerical": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
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

print(pipeline)
print(pipeline.search_space("configspace"))

"""
## Target Function
The function we will optimize must take in a `Trial` and return a `Trial.Report`.
We also pass in a [`PathBucket`][amltk.store.Bucket] which is a dict-like view of the
file system, where we have our dataset stored.

We also pass in our pipeline, which we will use to build our sklearn pipeline with a
specific `trial.config` suggested by the [`Optimizer`][amltk.optimization.Optimizer].
"""
from sklearn.metrics import accuracy_score

from amltk.optimization import Trial


def target_function(trial: Trial, _pipeline: Node) -> Trial.Report:
    trial.store({"config.json": trial.config})
    # Load in data
    with trial.profile("data-loading"):
        X_train, X_val, X_test, y_train, y_val, y_test = (
            trial.bucket["X_train.csv"].load(),
            trial.bucket["X_val.csv"].load(),
            trial.bucket["X_test.csv"].load(),
            trial.bucket["y_train.npy"].load(),
            trial.bucket["y_val.npy"].load(),
            trial.bucket["y_test.npy"].load(),
        )

    # Configure the pipeline with the trial config before building it.
    sklearn_pipeline = _pipeline.configure(trial.config).build("sklearn")

    # Fit the pipeline, indicating when you want to start the trial timing and error
    # catchnig.
    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

    # If an exception happened, we use `trial.fail` to indicate that the
    # trial failed
    if trial.exception:
        trial.store({"exception.txt": f"{trial.exception}\n {trial.traceback}"})
        return trial.fail()

    # Make our predictions with the model
    with trial.profile("predictions"):
        train_predictions = sklearn_pipeline.predict(X_train)
        val_predictions = sklearn_pipeline.predict(X_val)
        test_predictions = sklearn_pipeline.predict(X_test)

    with trial.profile("probabilities"):
        val_probabilites = sklearn_pipeline.predict_proba(X_val)

    # Save the scores to the summary of the trial
    with trial.profile("scoring"):
        train_acc = float(accuracy_score(train_predictions, y_train))
        val_acc = float(accuracy_score(val_predictions, y_val))
        test_acc = float(accuracy_score(test_predictions, y_test))

    trial.summary["train/acc"] = train_acc
    trial.summary["val/acc"] = val_acc
    trial.summary["test/acc"] = test_acc

    # Save all of this to the file system
    trial.store(
        {
            "model.pkl": sklearn_pipeline,
            "val_probabilities.npy": val_probabilites,
            "val_predictions.npy": val_predictions,
            "test_predictions.npy": test_predictions,
        },
    )

    # Finally report the success
    return trial.success(accuracy=val_acc)


"""
## Running the Whole Thing

Now we can run the whole thing. We will use the
[`Scheduler`][amltk.scheduling.Scheduler]
to run the optimization, and the
[`SMACOptimizer`][amltk.optimization.optimizers.smac.SMACOptimizer] to
optimize the pipeline.

### Getting and storing data
We use a [`PathBucket`][amltk.store.PathBucket] to store the data. This is a dict-like
view of the file system.
"""
from amltk.store import PathBucket

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

print(bucket)
print(dict(bucket))
"""
### Setting up the Scheduler, Task and Optimizer
We use the [`Scheduler.with_processes`][amltk.scheduling.Scheduler.with_processes]
method to create a [`Scheduler`][amltk.scheduling.Scheduler] that will run the
optimization.

Please check out the full [guides](../guides/index.md) to learn more!

We then create an [`SMACOptimizer`][amltk.optimization.optimizers.smac.SMACOptimizer]
which will optimize the pipeline. We pass in pipeline, and SMAC the optimizer will
parser out the space of hyperparameters to optimize.
"""
from amltk.optimization import Metric
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(2)

from amltk.optimization.optimizers.smac import SMACOptimizer

optimizer = SMACOptimizer.create(
    space=pipeline,  # <!> (1)!
    metrics=Metric("accuracy", minimize=False, bounds=(0.0, 1.0)),
    bucket=bucket,
    seed=seed,
)

# 1. You can also explicitly pass in the space of hyperparameters to optimize.
#   ```python
#   space = pipeline.search_space("configspace")
#   # or
#   space = pipeline.search_space(SMACOptimizer.preffered_parser())
#   ```
"""
Next we create a [`Task`][amltk.Task], passing in the function we
want to run and the scheduler we will run it in.
"""
task = scheduler.task(target_function)

print(task)
"""
We use the callback decorators of the [`Scheduler`][amltk.scheduling.Scheduler] and
the [`Task`][amltk.Task] to add callbacks that get called
during events that happen during the running of the scheduler. Using this, we can
control the flow of how things run.
Check out the [task guide](../guides/scheduling.md) for more.

This one here asks the optimizer for a new trial when the scheduler starts and
launches the task we created earlier with this trial.
"""


@scheduler.on_start
def launch_initial_tasks() -> None:
    """When we start, launch `n_workers` tasks."""
    trial = optimizer.ask()
    task.submit(trial, _pipeline=pipeline)


"""
When a [`Task`][amltk.Trial] returns and we get a report, i.e.
with [`task.success()`][amltk.optimization.Trial.success] or
[`task.fail()`][amltk.optimization.Trial.fail], the `task` will fire off the
callbacks registered with [`@on_result`][amltk.Task.on_result].
We can use these to add callbacks that get called when these events happen.

Here we use it to update the optimizer with the report we got.
"""


@task.on_result
def tell_optimizer(_, report: Trial.Report) -> None:
    """When we get a report, tell the optimizer."""
    optimizer.tell(report)


"""
We can use the [`History`][amltk.optimization.History] class to store the reports we get
from the [`Task`][amltk.Task]. We can then use this to analyze the results of the
optimization afterwords.
"""
from amltk.optimization import History

trial_history = History()


@task.on_result
def add_to_history(_, report: Trial.Report) -> None:
    """When we get a report, print it."""
    trial_history.add(report)


"""
We launch a new task when the scheduler is empty, i.e. when all the tasks have
finished. This will keep going until we hit the timeout we set on the scheduler.

If you want to run the optimization in parallel, you can use the
[`@task.on_result`][amltk.Task.on_result] callback to launch a new task when you get
a report. This will launch a new task as soon as one finishes.
"""

@task.on_result
def launch_another_task(*_: Any) -> None:
    """When we get a report, evaluate another trial."""
    if scheduler.running():
        trial = optimizer.ask()
        task.submit(trial, _pipeline=pipeline)


"""
If something goes wrong, we likely want to stop the scheduler.
"""


@task.on_exception
def stop_scheduler_on_exception(*_: Any) -> None:
    scheduler.stop()


@task.on_cancelled
def stop_scheduler_on_cancelled(_: Any) -> None:
    scheduler.stop()


"""
### Setting the system to run

Lastly we use [`Scheduler.run`][amltk.scheduling.Scheduler.run] to run the
scheduler. We pass in a timeout of 20 seconds.
"""
if __name__ == "__main__":
    scheduler.run(timeout=5)

    print("Trial history:")
    history_df = trial_history.df()
    print(history_df)
