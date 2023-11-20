# History
??? note "API links"

    * [`History`][amltk.optimization.History]
    * [`Report`][amltk.optimization.Trial.Report]

## Basic Usage
The [`History`][amltk.optimization.History] class is used to store
[`Report`][amltk.optimization.trial.Trial.Report]s from [`Trial`][amltk.optimization.Trial]s.

In it's most simple usage, you can simply [`add()`][amltk.optimization.History.add]
a `Report` as you recieve them and then use the [`df()`][amltk.optimization.History.df]
method to get a [`pandas.DataFrame`][pandas.DataFrame] of the history.

```python exec="true" source="material-block" result="python" title="Reference History" session="ref-history"
from amltk.optimization import Trial, History, Metric

loss = Metric("loss", minimize=True)

def quadratic(x):
    return x**2

history = History()
trials = [
    Trial(name=f"trial_{count}", config={"x": i}, metrics=[loss])
    for count, i in enumerate(range(-5, 5))
]

reports = []
for trial in trials:
    with trial.begin():
        x = trial.config["x"]
        report = trial.success(loss=quadratic(x))
        history.add(report)

print(history.df())
```

Typically, to use this inside of an optimization run, you would add the reports inside
of a callback from your [`Task`][amltk.Task]s. Please
see the [optimization guide](../../guides/optimization.md) for more details.

??? example "With an Optimizer and Scheduler"

    ```python
    from amltk.optimization import Trial, History, Metric
    from amltk.scheduling import Scheduler
    from amltk.pipeline import Searchable

    searchable = Searchable("quad", space={"x": (-5, 5)})
    n_workers = 2

    def quadratic(x):
        return x**2

    def target_function(trial: Trial) -> Trial.Report:
        x = trial.config["x"]
        with trial.begin():
            cost = quadratic(x)
            return trial.success(cost=cost)

    optimizer = SMACOptimizer(space=searchable, metrics=Metric("cost", minimize=True), seed=42)

    scheduler = Scheduler.with_processes(2)
    task = scheduler.task(quadratic)

    @scheduler.on_start(repeat=n_workers)
    def launch_trial():
        trial = optimizer.ask()
        task(trial)

    @task.on_result
    def add_to_history(report):
        history.add(report)

    @task.on_done
    def launch_another(_):
        trial = optimizer.ask()
        task(trial)

    scheduler.run(timeout=3)
    ```

### Querying
The [`History`][amltk.optimization.History] can be queried by either
an index or by the trial name.

```python exec="true" source="material-block" result="python" title="History Querying [str]" session="ref-history"
last_report = history[-1]
print(last_report)
print(history[last_report.name])
```

```python exec="true" source="material-block" result="python" session="ref-history"
for report in history:
    print(report.name, f"loss = {report.metrics['loss']}")
```

```python exec="true" source="material-block" result="python" session="ref-history"
sorted_history = history.sortby("loss")
print(sorted_history[0])
```

### Filtering
You can filter the history by using the [`filter()`][amltk.optimization.History.filter]
method. This method takes a [`Callable[[Trial.Report], bool]`][typing.Callable]
and returns a new [`History`][amltk.optimization.History] with only the
[`Report`][amltk.optimization.Trial.Report]s that return `True` from the
given function.

```python exec="true" source="material-block" result="python" title="Filtering" session="ref-history"
def is_even(report):
    return report.config["x"] % 2 == 0

even_history = history.filter(is_even)
even_history_df = even_history.df(profiles=False)
print(even_history_df)
```
