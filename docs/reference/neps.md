# NEPS

The below example shows how you can use neps to optimize an sklearn pipeline.

!!! todo "Deep Learning"

    Write an example demonstrating NEPS with continuations

!!! todo "Graph Search Spaces"

    Write an example demonstrating NEPS with its graph search spaces

```python
from __future__ import annotations

import logging

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from amltk import History, Pipeline, Trial, step
from amltk.neps import NEPSOptimizer, NEPSTrialInfo
from amltk.scheduling.scheduler import Scheduler

logging.basicConfig(level=logging.DEBUG)


def target_function(trial: Trial[NEPSTrialInfo], pipeline: Pipeline) -> Trial.Report:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = pipeline.configure(trial.config).build()

    with trial.begin():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        loss = 1 - accuracy
        return trial.success(loss=loss, accuracy=accuracy)

    return trial.fail()


pipeline = Pipeline.create(
    step("rf", RandomForestClassifier, space={"n_estimators": (10, 100)}),
)
optimizer = NEPSOptimizer.create(space=pipeline.space(), overwrite=True)


N_WORKERS = 4
scheduler = Scheduler.with_processes(N_WORKERS)
task = scheduler.task(target_function)

history = History()


@scheduler.on_start(repeat=N_WORKERS)
def on_start():
    trial = optimizer.ask()
    task.submit(trial, pipeline)


@task.on_result
def tell_and_launch_trial(_, report: Trial.Report):
    optimizer.tell(report)
    trial = optimizer.ask()
    task.submit(trial, pipeline)


@task.on_result
def add_to_history(_, report: Trial.Report):
    history.add(report)


scheduler.run(timeout=5, wait=False)

print(history.df())
history.to_csv("history.csv")
```
