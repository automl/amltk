# History
??? note "API links"

    * [`History`][byop.optimization.History]
    * [`Trace`][byop.optimization.Trace]
    * [`IncumbentTrace`][byop.optimization.IncumbentTrace]
    * [`Report`][byop.optimization.Trial.Report]

## Basic Usage
The [`History`][byop.optimization.History] class is used to store
[`Report`][byop.optimization.trial.Trial.Report]s from [`Trial`][byop.optimization.Trial]s.

In it's most simple usage, you can simply [`add()`][byop.optimization.History.add]
a `Report` as you recieve them and then use the [`df()`][byop.optimization.History.df]
method to get a [`pandas.DataFrame`][pandas.DataFrame] of the history.

```python exec="true" source="material-block" result="python" title="Reference History" session="ref-history"
from byop.optimization import Trial, History

def quadratic(x):
    return x**2

history = History()
trials = [
    Trial(f"trial_{count}", info=None, config={"x": i})
    for count, i in enumerate(range(-5, 5))
]

reports = []
for trial in trials:
    with trial.begin():
        cost = quadratic(trial.config["x"])
        report = trial.success(cost=cost)
        history.add(report)

print(history.df())
```

Typically, to use this inside of an optimization run, you would add the reports inside
of a callback from your [`Trial.Task`][byop.optimization.Trial.Task]s. Please
see the [optimization guide](../guides/optimization.md) for more details.

??? example "With an Optimizer and Scheduler"

    ```python
    from byop.optimization import RandomSearch, Trial, History, searchable
    from byop.scheduling import Scheduler

    search_space = searchable("quad", space={"x": (-5, 5)})
    n_workers = 2

    def quadratic(x):
        return x**2

    def target_function(trial: Trial) -> Trial.Report:
        x = trial.config["x"]
        with trial.begin():
            cost = quadratic(x)
            return trial.success(cost=cost)

    optimizer = RandomSearch(space=search_space.space(), seed=42)

    scheduler = Scheduler.with_processes(2)
    task = Trial.Task(quadratic)

    @scheduler.on_start(repeat=n_workers)
    def launch_trial():
        trial = optimizer.ask()
        task(trial)

    @task.on_report
    def add_to_history(report):
        history.add(report)

    @task.on_report
    def launch_another(_):
        trial = optimizer.ask()
        task(trial)

    scheduler.run(timeout=3)
    ```

## Querying
The [`History`][byop.optimization.History] acts like
a [`Mapping[str, Trial.Report]`][collections.abc.Mapping],
where the keys are the `trial.name` and the values are its
associated [`Report`][byop.optimization.Trial.Report]


```python exec="true" source="material-block" result="python" title="History as a Mapping" session="ref-history"
print(history["trial_0"])

print("iterating")
for name, report in history.items():
    print(name, f"cost = {report.results['cost']}")

print(f"len = {len(history)}")

sorted_reports = sorted(history.values(), key=lambda r: r.results['cost'])
print(f"best = {sorted_reports[0]}")
```

## Filtering
You can filter the history by using the [`filter()`][byop.optimization.History.filter]
method. This method takes a [`Callable[[Trial.Report], bool]`][typing.Callable]
and returns a new [`History`][byop.optimization.History] with only the
[`Report`][byop.optimization.Trial.Report]s that return `True` from the
given function.

```python exec="true" source="material-block" result="python" title="Filtering" session="ref-history"
def is_even(report):
    return report.config["x"] % 2 == 0

even_history = history.filter(is_even)
even_history_df = even_history.df()
print(even_history_df[["results:cost", "config:x"]])
```

# Traces
A [`Trace`][byop.optimization.Trace] is just the history
but ordered in some particular way. This means it acts `list`-like,
specifically a [`Sequence[Trial.Report]`][collections.abc.Sequence], meaning
it has an order, we can index it and we can iterate over it.

We obtain a [`Trace`][byop.optimization.Trace] by using
the [`sortby()`][byop.optimization.History.sortby] method to
order the [`History`][byop.optimization.History] in some way.

```python exec="true" source="material-block" result="python" title="Trace" session="ref-history"
from byop.optimization import Trace

trace = history.sortby(lambda report: report.time.end)
print(trace.df())
```

To use it like a list:

```python exec="true" source="material-block" result="python" title="Trace as a Sequence" session="ref-history"
for report in trace[:2]:
    print(report)

print(f"len = {len(trace)}")

print(f"first = {trace[0]}")
print(f"last = {trace[-1]}")

sorted_reports = sorted(trace, key=lambda r: r.results['cost'])
print(f"best = {sorted_reports[0]}")
```

# Incumbent Trace
An [`IncumbentTrace`][byop.optimization.IncumbentTrace] is a
[`Trace`][byop.optimization.Trace] that only contains the
**best** [`Trial.Report`][byop.optimization.Trial.Report]s
along a given axis. For example, if we have
the numbers `#!python [10, 8, 5, 7, 7, 2, 8]` and we define
the **best** as the smallest number, then the
an incumbany trace would look like `#!python [10, 8, 5, 2]`.

We can do the same for our [`Trace`][byop.optimization.Trace]
objects by using the [`incumbents()`][byop.optimization.Trace.incumbents]
method.

```python exec="true" source="material-block" result="python" title="Incumbent Trace" session="ref-history"
import operator

trace = history.sortby(lambda report: report.time.end)
incumbent_trace = trace.incumbents(lambda report: report.results["cost"])

print(incumbent_trace.df())
```

!!! note "Defining Best"

    By default, the [`incumbents()`][byop.optimization.Trace.incumbents]
    method uses the [`lt`][operator.lt] which says that the **best**
    is the smallest value. You can change this by passing in a different
    [`Callable[[Any, Any], bool]`][typing.Callable] as the `op` argument.
    For example, say we wanted to take the **best** to be the largest
    value, we could do the following:

    ```python exec="true" source="material-block" result="python" title="Incumbent Trace (max)" session="ref-history"
    import operator

    trace = history.sortby(lambda report: report.time.end)
    incumbent_trace = trace.incumbents(lambda report: report.results["cost"], op=operator.gt)

    print(incumbent_trace.df())
    ```
