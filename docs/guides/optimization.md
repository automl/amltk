# Optimization Guide
One of the core tasks of any AutoML system is to optimize some objective,
whether it be some pipeline, a black-box or even a toy function. In the context
of AMLTK, this means defining some [`Metric(s)`](../reference/optimization/metrics.md) to optimize
and creating an [`Optimizer`](../reference/optimization/optimizers.md) to optimize
them.

You can check out the integrated optimizers in our [optimizer reference](../reference/optimization/optimizers.md)


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
we can reprsent entire machine learning pipelines, with conditonality and much more complex ranges. ([Pipeline guide](../guides/pipelines.md))

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
from amltk._doc import doc_print; doc_print(print, s)
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

The first thing we'll need to do is create a [`Metric`](../reference/optimization/metrics.md)
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
```

## Running an Optimizer
At this point, we can begin optimizing our function, using the [`ask`][amltk.optimization.Optimizer.ask]
to get [`Trial`][amltk.optimization.Trial]s and [`tell`][amltk.optimization.Optimizer.tell] methods with
[`Trial.Report`][amltk.optimization.Trial.Report]s.

```python exec="true" result="python" source="material-block"
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.optimization import Metric, History, Trial
from amltk.pipeline import Searchable

def poly(x: float) -> float:
    return (x**2 + 4*x + 3) / x

metric = Metric("score", minimize=False)
space = Searchable(space={"x": (-10.0, 10.0)}, name="my-searchable")

optimizer = SMACOptimizer.create(space=space, metrics=metric, seed=42)

reports = []
for _ in range(10):
    trial: Trial = optimizer.ask()
    print(f"Evaluating trial {trial.name} with config {trial.config}")
    x = trial.config["my-searchable:x"]

    with trial.begin():
        score = poly(x)

    report: Trial.Report = trial.success(score=score)
    reports.append(report)

last_report = reports[-1]
print(last_report.df())
optimizer.bucket.rmdir()  # markdown-exec: hide
```

Okay so there are a few things introduced all at once here, let's go over them bit by bit.

### The `Trial` object
The [`Trial`](../reference/optimization/trials.md) object is the main object that
you'll be interacting with when optimizing. It contains a load of useful properties and
functionality to help you during optimization. What we introduced here is the `.config`,
which contains a key, value mapping of the parameters to optimize, in this case, `x`. We
also wrap the actual evaluation of the function in a
[`with trial.begin():`][amltk.optimization.Trial.begin] which will time and profile the
evaluation of the function and handle any exceptions that occur within the block, attaching
the exception to [`.exception`][amltk.optimization.Trial.exception] and the traceback to
[`.traceback`][amltk.optimization.Trial.traceback]. Lastly, we use
[`trial.success()`][amltk.optimization.Trial.success] which generates a [`Trial.Report`][amltk.optimization.Trial.Report]
for us.

We'll cover more of this later in the guide but feel free to check out the full [API][amltk.optimization.Trial].


---

!!! todo "TODO"

    Everything past here is likely out-dated, sorry. Matrial
    in the [pipelines guide](./pipelines.md) guide and the
    [scheduling guide](./scheduling.md) is more up-to-date.

## Running an Optimizer
Now that we have an optimizer that knows the `space` to search, we can begin to
actually [`ask()`][amltk.optimization.Optimizer.ask] the optimizer for a next
[`Trial`][amltk.optimization.Trial], run our function and return
a [`Trial.Report`][amltk.optimization.Trial.Report].

First we need to modify our function we wish to optimize to actually accept
the `Trial` and return the `Report`.

```python hl_lines="4 5 6 7 8 9 10 19 20 21 22 24 25" title="Runnig the Optimizer"
from amltk.optimization import RandomSearch, Trial
from amltk.pipeline import searchable

def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:  # (4)!
    x = trial.config["x"]
    with trial.begin():  # (1)!
        y = (x**2 + 4*x + 3) / x
        return trial.success(cost=y)  # (2)!

    trial.fail()  # (3)!

s = searchable("parameters", space={"x": (-10.0, 10.0)})

space = s.space()
random_search = RandomSearch(space=space, seed=42)

results: list[float] = []

for _ in range(20):
    trial = random_search.ask()
    report = qaudratic(trial)
    random_search.tell(trial)

    cost = report.results["cost"]
    results.append(cost)
```

1. Using the [`with trial.begin():`][amltk.optimization.Trial.begin],
you let us know where exactly your trial begins and we can handle
all things related to exception handling and timing.
2. If you can return a success, then do so with
[`trial.success()`][amltk.optimization.Trial.success].
3. If you can't return a success, then do so with [`trial.fail()`][amltk.optimization.Trial.fail].
4. Here the inner type parameter `RSTrial` is the type of `trial.info` which
contains the object returned by the ask of the wrapped `optimizer`. We'll
see this in [integrating your own Optimizer](#integrating-your-own-optimizer).

### Running the Optimizer in a parallel fashion

Now that we've seen the basic optimization loop, it's time to parallelize it with
a [`Scheduler`][amltk.scheduling.Scheduler] and the [`Task`][amltk.Task].
We cover the [`Scheduler`][amltk.scheduling.Scheduler] and [`Tasks`][amltk.scheduling.Task]
in the [Scheduling guide](./scheduling.md) if you'd like to know more about how this works.

We first create a [`Scheduler`][amltk.scheduling.Scheduler] to run with `#!python 1`
process and run it for `#!python 5` seconds.
Using the event system of AutoML-Toolkit,
we define what happens through _callbacks_, registering to certain events, such
as launch a single trial on `@scheduler.on_start`, _tell_ the optimizer whenever we get
something returned with [`@task.on_result`][amltk.Task.on_result].


```python hl_lines="19 23 24 25 26 28 29 30 32 33 34 35 37 38 39 40 42" title="Creating a Task for a Trial"
from amltk.optimization import RandomSearch, Trial, RSTrialInfo
from amltk.pipeline import searchable
from amltk.scheduling import Scheduler

def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:
    x = trial.config["x"]
    with trial.begin():
        y = (x**2 + 4*x + 3) / x
        return trial.success(cost=y)

    trial.fail()

s = searchable("parameters", space={"x": (-10.0, 10.0)})
space = s.space()

random_search = RandomSearch(space=space, seed=42)
scheduler = Scheduler.with_processes(1)

task = scheduler.task(poly)  # (5)!

results: list[float] = []

@scheduler.on_start  # (1)!
def launch_trial() -> None:
    trial = random_search.ask()
    task(trial)

@task.on_result  # (2)!
def tell_optimizer(report: Trial.Report) -> None:
    random_search.tell(report)

@task.on_result
def launch_another_trial(_: Trial.Report) -> None:
    trial = random_search.ask()
    task(trial)

@task.on_result  # (3)!
def save_result(report: Trial.Report) -> None:
    cost = report["cost"]
    results.append(cost)  # (4)!

scheduler.run(timeout=5)
```

1. The function `launch_trial()` gets called when the `scheduler` starts,
asking the optimizer for a trial and launching the `task` with the `trial`.
`launch_trial()` gets called in the main process but `task(trial)` will get
called in a seperate process.
2. The function `tell_optimizer` gets called whenever the `task` returns a
report. We should tell the optimizer about this report.
3. This function `save_result` gets called whenever we have a successful
trial.
4. We don't store anything more than the optmimizer needs. Saving results
that you wish to access later is up to you.
5. Here we wrap the function we want to run in another process in a
[`Task`][amltk.optimization.Trial]. There are other backends than
processes, e.g. Clusters for which you should check out the
[Scheduling guide](./scheduling.md).

Now, to scale up, we trivially increase the number of initial trails launched with `@scheduler.on_start`
and the number of processes in our `Scheduler`. That's it.

```python hl_lines="18 19 25"
from amltk.optimization import RandomSearch, Trial, RSTrialInfo
from amltk.pipeline import searchable
from amltk.scheduling import Scheduler

def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:
    x = trial.config["x"]
    with trial.begin():
        y = (x**2 + 4*x + 3) / x
        return trial.success(cost=y)

    trial.fail()

s = searchable("parameters", space={"x": (-10.0, 10.0)})
space = s.space()

random_search = RandomSearch(space=space, seed=42)

n_workers = 4
scheduler = Scheduler.with_processes(n_workers)

task = Trial.Task(poly)

results: list[float] = []

@scheduler.on_start(repeat=n_workers)
def launch_trial() -> None:
    trial = random_search.ask()
    task(trial)

@task.on_result
def tell_optimizer(report: Trial.Report) -> None:
    random_search.tell(report)

@task.on_result
def launch_another_trial(_: Trial.Report) -> None:
    trial = random_search.ask()
    task(trial)

@task.on_result
def save_result(report: Trial.Report) -> None:
    cost = report["cost"]
    results.append(cost)

scheduler.run(timeout=5)
```

That concludes the main portion of our `Optimization` guide. AutoML-Toolkit provides
a host of more useful options, such as:

* Setting constraints on your evaluation function, such as memory, wall time and cpu time, concurrency limits
and call limits. Please refer to the [Scheduling guide](./scheduling.md) for more information.
* Stop the scheduler with whatever stopping criterion you wish. Please refer to the [Scheduling guide](./scheduling.md) for more information.
* Optimize over complex pipelines. Please refer to the [Pipeline guide](./pipelines.md) for more information.
* Using different parallelization strategies, such as [Dask](https://dask.org/), [Ray](https://ray.io/),
[Slurm](https://slurm.schedmd.com/), and [Apache Airflow](https://airflow.apache.org/).
* Use a whole host of more callbacks to control you system, check out the [Scheduling guide](./scheduling.md) for more information.
* Run the scheduler using `asyncio` to allow interactivity, run as a server or other more advanced use cases.
