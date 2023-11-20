[![image](https://img.shields.io/pypi/v/amltk.svg)](https://pypi.python.org/pypi/amltk)
[![image](https://img.shields.io/pypi/l/amltk.svg)](https://pypi.python.org/pypi/amltk)
[![image](https://img.shields.io/pypi/pyversions/amltk.svg)](https://pypi.python.org/pypi/amltk)
[![Actions](https://github.com/automl/amltk/actions/workflows/test.yml/badge.svg)](https://github.com/automl/amltk/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AutoML Toolkit
A framework for defining, configuring, building and evaluating AutoML pipelines.
Please check out our documentation:
* [Documentation](https://automl.github.io/amltk/)

## Features

### Pipelines
Define **parametrized** machine learning pipelines using a fluid API:
```python
from amltk.pipeline import Component, Choice, Sequential
from sklearn.ensemble import RandomForestClasifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

pipeline = (
    Sequential(name="my_pipeline")
    >> Component(SimpleImputer, space={"strategy": ["mean", "median"]}),  # Choose either mean or median
    >> OneHotEncoder(drop="first")  # No parametrization, no problem
    >> Choice(
        # Our pipeline can choose between two different estimators
        Component(
            RandomForestClassifier,
            space={
                "n_estimators": (10, 100),
                "criterion": ["gini", "log_loss"]
            },
            config={"max_depth":3}
        ),
        Component(SVC, space={"kernel": ["linear", "rbf", "poly"]}),
        name="estimator"
    )
)

# Parser the search space with implemented or you custom parser
search_space = pipeline.search_space(parser=...)

# Configure a pipeline
configured_pipeline = pipeline.configure(config)

# Build the pipeline with a build, no amltk code in your built model
model = configured_pipeline.build(builder="sklearn")
```

### Optimizers
Optimize your pipelines using a variety of different optimizers, with a unified API and
a suite of utility for recording and taking control of the optimization loop:

```python
from amltk.optimization import Trial, Metric, History

pipeline = ...
accuracy = Metric("accuracy", maximize=True, bounds=(0. 1))
inference_time = Metric("inference_time", maximize=False)

def evaluate(trial: Trial) -> Trial.Report:

    # Say when and where you trial begins
    with trial.begin():
        model = pipeline.configure(trial.config).build("sklearn")

        # Profile the things you'd like
        with trial.profile("fit"):
            model.fit(...)

        # Record anything else you'd like
        trial.summary["model_size"] = ...

        # Store whatever you'd like
        trial.store({"model.pkl": model, "predictions.npy": predictions}),
        return trial.success(accuracy=0.8, inference_time=...)

    if trial.exception:
        return trial.fail()

# Easily swap between optimizers, without needing to change the rest of your code
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.optimization.optimizers.smac import OptunaOptimizer
import random

Optimizer = random.choice([SMACOptimizer, OptunaOptimizer])
smac_optimizer = Optimizer(space=pipeline, metrics=[accuracy, inference_time], bucket="results")


# You decide how your optimization loop should work
history = History()
for _ in range(10):
    trial = optimizer.ask()
    report = evaluate(trial)
    history.add(report)
    optimizer.tell(report)

print(history.df())
```

> [!TIP]
> Check out our [integrated optimizers](https://automl.github.io/amltk/latest/reference/optimization/optimizers) or integrate your own using the very
> same API we use!

### Scheduling
Schedule your optimization jobs or AutoML tasks on a variety of different compute backends. By leveraging
compute workers and asyncio, you can easily scale your compute needs, react to events as they happen and
swap backends, without needing to modify your code!

```python
from amltk.scheduling import Scheduler

# Create a Scheduler with a backend, here 4 processes
scheduler = Scheduler.with_processes(4)

# Define some compute and wrap it as a task to offload to the scheduler
def expensive_function(x: int) -> float:
    return (2 ** x) / x

task = scheduler.task(expensive_function)

numbers = range(-5, 5)
results = []

# When the scheduler starts, submit 4 tasks to the processes
@scheduler.on_start(repeat=4)
def on_start():
    n = next(numbers)
    task.submit(n)

# When the task is done, store the result
@task.on_result
def on_result(_, result: float):
    results.append(result)

# Easy to incrementently add more functionallity
@task.on_result
def launch_next(_, result: float):
    if (n := next(numbers, None)) is not None:
        task.submit(n)

# React to issues when they happen
@task.on_exception
def stop_something_went_wrong(_, exception: Exception):
    scheduler.stop()

# Start the scheduler and run it as you like
scheduler.run(timeout=10)

# ... await scheduler.async_run() for servers and real-time applications
```

> [!TIP]
> Check out our [integrated compute backends](https://automl.github.io/amltk/latest/reference/scheduling/executors) or use your own!


### Installation
To install from source, you can clone this repo and install with `pip`:

```bash
git clone git@github.com:automl/amltk.git
pip install amltk
```

> [!TIP]
> We have some optional dependancies you can install too:
> * `pip install amltk[notebook]` - For usage in a notebook
> * `pip install amltk[sklearn]` - For usage with scikit-learn
> * `pip install amltk[smac]` - For using SMAC as an optimizer
> * `pip install amltk[optuna]` - For using Optuna as an optimizer
> * `pip install amltk[pynisher,  threadpoolctl, wandb]` - Various plugins for running compute tasks
> * `pip install amltk[cluster, dask, loky]` - Different compute backends to run from
