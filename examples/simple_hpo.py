"""Performing HPO with Post-Hoc Ensembling.
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn]`

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
from pathlib import Path
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

from byop.optimization import History, Trial
from byop.pipeline import Pipeline, split, step
from byop.scheduling import Scheduler
from byop.sklearn.data import split_data
from byop.smac import SMACOptimizer
from byop.store import PathBucket

"""
Below is just a small function to help us get the dataset from OpenML
and encode the labels.
"""


def get_dataset(seed: int, splits: dict[str, float]) -> dict[str, Any]:
    dataset = openml.datasets.get_dataset(31)

    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)

    return split_data(X, _y, splits=splits, seed=seed)  # type: ignore


"""
## Pipeline Definition

Here we define a pipeline which splits categoricals and numericals down two
different paths, and then combines them back together before passing them to
a choice of classifier between a Random Forest, Support Vector Machine, and
Multi-Layer Perceptron.

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
            "n_estimators": [10, 100],
            "criterion": ["gini", "entropy", "log_loss"],
        },
    ),
)

print(pipeline)
print(pipeline.space())


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

    with trial.begin():
        sklearn_pipeline.fit(X_train, y_train)

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
    val_accuracy = accuracy_score(val_predictions, y_val)

    # Save the scores to the summary of the trial
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

    return trial.success(cost=1 - val_accuracy)


seed = 42
data = get_dataset(seed, splits={"train": 0.6, "val": 0.2, "test": 0.2})
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

scheduler = Scheduler.with_sequential()
optimizer = SMACOptimizer.HPO(space=pipeline.space(), seed=seed)

objective = Trial.Objective(
    target_function,
    bucket=bucket,
    pipeline=pipeline,
)
task = Trial.Task(objective, scheduler)
trial_history = History()


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


@task.on_report
def launch_another_task(_: Trial.Report) -> None:
    """When we get a report, evaluate another trial."""
    trial = optimizer.ask()
    task(trial)

scheduler.run(timeout=5, wait=True)

print("Trial history:")
history_df = trial_history.df()
print(history_df)
