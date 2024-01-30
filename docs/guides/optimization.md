# Optimization Guide
One of the core tasks of any AutoML system is to optimize some objective,
whether it be some pipeline, a black-box or even a toy function. In the context
of AMLTK, this means defining some [`Metric(s)`](../reference/optimization/metrics.md) to optimize
and creating an [`Optimizer`](../reference/optimization/optimizers.md) to optimize
them.

You can check out the integrated optimizers in our [optimizer reference](../reference/optimization/optimizers.md).


This guide relies lightly on topics covered in the [Pipeline Guide](../guides/pipelines.md) for
creating a pipeline but also the [Scheduling guide](../guides/scheduling.md) for creating a
[`Scheduler`][amltk.scheduling.Scheduler] and a [`Task`][amltk.scheduling.Task].
These aren't required but if something is not clear or you'd like to know **how** something
works, please refer to these guides or the reference!


## Optimizing a 1-D function
We'll start with a simple example of **maximizing** a polynomial function
The first thing to do is define the function we want to optimize.

```python exec="true" source="material-block" html="true"
import numpy as np
import matplotlib.pyplot as plt

def poly(x):
    return (x**2 + 4*x + 3) / x

fig, ax = plt.subplots()
x = np.linspace(-10, 10, 100)
ax.plot(x, poly(x))
from io import StringIO; fig.tight_layout(); buffer = StringIO(); plt.savefig(buffer, format="svg"); print(buffer.getvalue())  # markdown-exec: hide
```

Our next step is to define the search range over which we want to optimize, in
this case, the range of values `x` can take. Here we use a simple [`Searchable`][amltk.pipeline.Searchable], however
we can represent entire machine learning pipelines, with conditionality and much more complex ranges. ([Pipeline guide](../guides/pipelines.md))

!!! info inline end "Vocab..."

    When dealing with such functions, one might call the `x` just a parameter. However in
    the context of Machine Learning, if this `poly()` function was more like `train_model()`,
    then we would refer to `x` as a _hyperparameter_ with it's range as it's _search space_.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Searchable

def poly(x: float) -> float:
    return (x**2 + 4*x + 3) / x

s = Searchable(
    {"x": (-10.0, 10.0)},
    name="my-searchable"
)
from amltk._doc import doc_print; doc_print(print, s)  # markdown-exec: hide
```


## Creating an Optimizer

We'll utilize [SMAC](https://github.com/automl/SMAC3) here for optimization as an example
but you can find other available optimizers [here](../reference/optimization/optimizers.md).

??? info inline end "Requirements"

    This requires `smac` which can be installed with:

    ```bash
    pip install amltk[smac]

    # Or directly
    pip install smac
    ```

The first thing we'll need to do is create a [`Metric`](../reference/optimization/metrics.md):
a definition of some value we want to optimize.

```python exec="true" result="python" source="material-block"
from amltk.optimization import Metric

metric = Metric("score", minimize=False)
print(metric)
```

The next step is to actually create an optimizer, you'll have to refer to their
[reference documentation](../reference/optimization/optimizers.md). However, for most integrated optimizers,
we expose a helpful [`create()`][amltk.optimization.optimizers.smac.SMACOptimizer.create].

```python exec="true" result="python" source="material-block"
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.optimization import Metric
from amltk.pipeline import Searchable

def poly(x: float) -> float:
    return (x**2 + 4*x + 3) / x

metric = Metric("score", minimize=False)
space = Searchable(space={"x": (-10.0, 10.0)}, name="my-searchable")

optimizer = SMACOptimizer.create(space=space, metrics=metric, seed=42)
optimizer.bucket.rmdir()  # markdown-exec: hide
```

## Running an Optimizer
At this point, we can begin optimizing our function, using the [`ask`][amltk.optimization.Optimizer.ask]
to get [`Trial`][amltk.optimization.Trial]s and [`tell`][amltk.optimization.Optimizer.tell] methods with
[`Trial.Report`][amltk.optimization.Trial.Report]s.

```python exec="true" result="python" source="material-block" session="running-an-optimizer"
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.optimization import Metric, History, Trial
from amltk.pipeline import Searchable

def poly(x: float) -> float:
    return (x**2 + 4*x + 3) / x

metric = Metric("score", minimize=False)
space = Searchable(space={"x": (-10.0, 10.0)}, name="my-searchable")

optimizer = SMACOptimizer.create(space=space, metrics=metric, seed=42)

history = History()
for _ in range(10):
    # Get a trial from an Optimizer
    trial: Trial = optimizer.ask()
    print(f"Evaluating trial {trial.name} with config {trial.config}")

    # Access the the trial's config
    x = trial.config["my-searchable:x"]

    try:
        score = poly(x)
    except ZeroDivisionError as e:
        # Generate a failed report (i.e. poly(x) raised divide by zero exception with x=0)
        report = trial.fail(e)
    else:
        # Generate a success report
        report = trial.success(score=score)

    # Store artifacts with the trial, using file extensions to infer how to store it
    trial.store({ "config.json": trial.config, "array.npy": [1, 2, 3] })

    # Tell the Optimizer about the report
    optimizer.tell(report)

    # Add the report to the history
    history.add(report)
optimizer.bucket.rmdir()  # markdown-exec: hide
```

And we can use the [`History`][amltk.optimization.History] to see the history of the optimization
process

```python exec="true" result="python" source="material-block" session="running-an-optimizer"
df = history.df()
print(df)
```

Okay so there are a few things introduced all at once here, let's go over them bit by bit.

### The `Trial` object
The [`Trial`](../reference/optimization/trials.md) object is the main object that
you'll be interacting with when optimizing. It contains a load of useful properties and
functionality to help you during optimization.

The `.config` will contain name spaced parameters, in this case, `my-searchable:x`, based on the
pipeline/search space you specified.

It's also quite typical to store artifacts with the trial, a common feature of things like TensorBoard, MLFlow, etc.
We provide a primitive way to store artifacts with the trial using [`.store()`][amltk.optimization.Trial.store] which
takes a dictionary of file names to file contents. The file extension is used to infer how to store the file, for example,
`.json` files will be stored as JSON, `.npy` files will be stored as numpy arrays. You are of course still free to use
your other favourite logging tools in conjunction with AMLTK!

Lastly, we use [`trial.success()`][amltk.optimization.Trial.success] or [`trial.fail()`][amltk.optimization.Trial.fail]
which generates a [`Trial.Report`][amltk.optimization.Trial.Report] for us, that we can give back to the optimizer.

Feel free to explore the full [API][amltk.optimization.Trial].

### The `History` object
You may have noticed that we also created a [`History`][amltk.optimization.History] object to store our reports in. This
is a simple container to store the reports together and get a dataframe out of. We may extend this with future utility
such as plotting or other export formats but for now, we can use it primarily for getting our results together in one
place.

We'll create a simple example where we create _our own trials_ and record some results on them, getting out a dataframe
at the end.

```python exec="true" result="python" source="material-block"
from amltk.optimization import History, Trial, Metric
from amltk.store import PathBucket

metric = Metric("score", minimize=False, bounds=(0, 5))
history = History()

trials = [
    Trial.create(name="trial-1", config={"x": 1.0}, metrics=[metric]),
    Trial.create(name="trial-2", config={"x": 2.0}, metrics=[metric]),
    Trial.create(name="trial-3", config={"x": 3.0}, metrics=[metric]),
]

for trial in trials:
    x = trial.config["x"]
    if x >= 2:
        report = trial.fail()
    else:
        report = trial.success(score=x)

    history.add(report)

df = history.df()
print(df)

best = history.best()
print(best)
for t in trials: t.bucket.rmdir()  # markdown-exec: hide
```

You can use the [`History.df()`][amltk.optimization.History.df] method to get a dataframe of the history and
use your favourite dataframe tools to analyze the results.

## Optimizing an Sklearn-Pipeline
To give a more concrete example, we will optimize a simple sklearn pipeline. You'll likely want to refer to the
[pipeline guide](../guides/pipelines.md) for more information on pipelines, but the example should be clear
enough without it.

We start with defining our pipeline.

```python exec="true" html="true" source="material-block" session="optimizing-an-sklearn-pipeline"
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier

from amltk.pipeline import Sequential, Choice, Component

def dims_to_hidden_layer(config: dict[str, Any], _):
    config = dict(config)
    config["hidden_layer_sizes"] = (config.pop("dim1"), config.pop("dim2"))
    return config

# A pipeline with a choice of scalers and a parametrized MLP
my_pipeline = (
    Sequential(name="my-pipeline")
    >> Choice(
        StandardScaler,
        MinMaxScaler,
        Component(RobustScaler, space={"with_scaling": [True, False], "unit_variance": [True, False]}),
        name="scaler",
    )
    >> Component(
        MLPClassifier,
        space={
            "dim1": (1, 10),
            "dim2": (1, 10),
            "activation": ["relu", "tanh", "logistic"],
        },
        config_transform=dims_to_hidden_layer,
    )
)
from amltk._doc import doc_print; doc_print(print, my_pipeline)  # markdown-exec: hide
```

Next up, we need to define a simple target function we want to evaluate on.

```python exec="true" result="python" source="material-block" session="optimizing-an-sklearn-pipeline"
from sklearn.model_selection import cross_validate
from amltk.optimization import Trial
from amltk.store import Stored
import numpy as np

def evaluate(
    trial: Trial,
    pipeline: Sequential,
    X: Stored[np.ndarray],
    y: Stored[np.ndarray],
) -> Trial.Report:
    # Configure our pipeline and build it
    sklearn_pipeline = (
        pipeline
        .configure(trial.config)
        .build("sklearn")
    )

    # Load in our data
    X = X.load()
    y = y.load()

    # Use sklearns.cross_validate as our evaluator
    with trial.profile("cross-validate"):
        results = cross_validate(sklearn_pipeline, X, y, scoring="accuracy", cv=3, return_estimator=True)

    test_scores = results["test_score"]
    estimators = results["estimator"]  # You can store these if you like (you'll likely want to use the `.pkl` suffix for the filename)

    # Report the mean test score
    mean_test_score = np.mean(test_scores)
    return trial.success(acc=mean_test_score)
```

With that, we'll also store our data, so that on each evaluate call, we load it in.
This doesn't make much sense for a single in-process call but when scaling up to using
multiple processes or remote compute, this is a good practice to follow.

For this we use a [`PathBucket`][amltk.store.PathBucket] and get
a [`Stored`][amltk.store.Stored] from it, a reference to some object we can `load()` back in later.

```python exec="true" result="python" source="material-block" session="optimizing-an-sklearn-pipeline"
from sklearn.datasets import load_iris
from amltk.store import PathBucket

# Load in our data
_X, _y = load_iris(return_X_y=True)

# Store our data in a bucket
bucket = PathBucket("my-bucket")
stored_X = bucket["X.npy"].put(_X)
stored_y = bucket["y.npy"].put(_y)
```

Lastly, we'll create our optimizer and run it.
In this example, we'll use the [`SMACOptimizer`][amltk.optimization.optimizers.smac.SMACOptimizer] but
you can refer to the [optimizer reference](../reference/optimization/optimizers.md) for other optimizers. For basic
use cases, you should be able to swap in and out the optimizer and it should work without any changes.

```python exec="true" result="python" source="material-block" session="optimizing-an-sklearn-pipeline"
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.optimization import Metric, History

metric = Metric("acc", minimize=False, bounds=(0, 1))
optimizer = SMACOptimizer.create(
    space=my_pipeline,  # Let it know what to optimize
    metrics=metric,  # And let it know what to expect
    bucket=bucket,  # And where to store artifacts for trials and optimizer output
)

history = History()

for _ in range(10):
    # Get a trial from the optimizer
    trial = optimizer.ask()

    # Evaluate the trial
    report = evaluate(trial=trial, pipeline=my_pipeline, X=stored_X, y=stored_y)

    # Tell the optimizer about the report
    optimizer.tell(report)

    # Add the report to the history
    history.add(report)

df = history.df()
optimizer.bucket.rmdir()  # markdown-exec: hide
print(df)  # markdown-exec: hide
```
