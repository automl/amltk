One of the core tasks of any AutoML system is to optimize some objective,
whether it be some [`Pipeline`][byop.pipeline.Pipeline], a black-box or even a toy function.

For this we require an [`Optimizer`][byop.optimization.Optimizer]. We integrate
several optimizers and integrating your own is very straightforward, under one
very central premise.

!!! info "Premise: Ask-And-Tell"

    We explicitly require optimizers to support an _"Ask-and-Tell"_ interface.
    This means we can _"Ask"_ and optimizer for its next suggested configuration
    and we can _"Tell"_ it a result when we have one. **In fact, here's the entire
    interface.**

    ```python
    class Optimizer(Protocol):

        def tell(self, report: Trial.Report) -> None: ...

        def ask(self) -> Trial: ...

    ```

??? note "Why?"

    1. **Easy Parallelization**: Many optimizers handle running the function to optimize and hence roll out their own
    parallelization schemes and store data in all various different ways.

    2. **API maintenance**: Many optimziers are also unstable with respect to their API so wrapping around them can be difficult.
    By requiring this _"Ask-and-Tell"_ interface, we reduce the complexity of what is required
    of both the "Optimizer" and wrapping it.

    2. **Full Integration**: We can fully hook into the life cycle of a running optimizer. We are not relying
    on the optimizer to support callbacks at every step of their _hot-loop_ and as such, we
    can fully leverage all the other systems of AutoML-toolkit

    4. **Easy Integration**: it makes developing and integrating new optimizers easy. You only have
    to worry that the internal state of the optimizer is updated accordingly to these
    two _"Ask"_ and _"Tell"_ events and that's it.

This guide relies lightly on topics covered in the [Pipeline Guide](./pipelines.md) for
creating a `Pipeline` but also the [Task guide](./tasks.md) for creating a
[`Scheduler`][byop.scheduling.Scheduler] and a [`Task`][byop.scheduling.Task].
These aren't required but if something is not clear or you'd like to know **how** something
works, please refer to these guides


## Optimizating a simple function
We'll start with a simple example of optimizing a simple polynomial function
The first thing to do is define the function we want to optimize.

=== "Polynomial"

    ```python
    def poly(x):
        return (x**2 + 4*x + 3) / x
    ```

=== "Typed"

    ```python
    def poly(x: float) -> float:
        return (x**2 + 4*x + 3) / x
    ```

Our next step is to define the search range over which we want to optimize, in
this case, the range of values `x` can take. We cover this in more detail
in the [Pipeline guide](./pipelines.md).

=== "Defining a Search Space"

    ```python hl_lines="6"
    from byop.pipeline import searchable

    def poly(x):
        return (x**2 + 4*x + 3) / x

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})  # (1)!
    ```

    1. Here we say that there is a collection of `#!python "parameters"`
    which has one called `#!python "x"` which is in the range `#!python [-10.0, 10.0]`.

=== "Typed"

    ```python hl_lines="6"
    from byop.pipeline import searchable

    def poly(x: float) -> float:
        return (x**2 + 4*x + 3) / x

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})  # (1)!
    ```

    1. Here we say that there is a collection of `#!python "parameters"`
    which has one called `#!python "x"` which is in the range `#!python [-10.0, 10.0]`.

### Creating an optimizer

We'll start by using [`RandomSearch`][byop.optimization.RandomSearch] to search
for an optimal value for `#!python "x"` but later on we'll switch to using
[SMAC](https://github.com/automl/SMAC3) which is a much smarter optimizer.

=== "Creating an optmizer"

    ```python hl_lines="9 10"
    from byop.optimization import RandomSearch
    from byop.pipeline import searchable

    def poly(x):
        return (x**2 + 4*x + 3) / x

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})

    space = s.space()
    random_search = RandomSearch(space=space, seed=42)
    ```

=== "Typed"

    ```python hl_lines="9 10"
    from byop.optimization import RandomSearch
    from byop.pipeline import searchable

    def poly(x: float) -> float:
        return (x**2 + 4*x + 3) / x

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})

    space = s.space()
    random_search = RandomSearch(space=space, seed=42)
    ```

Some of the integrated optimizers:

---

- __Random Search__

    A custom implementation of Random Search which randomly selects
    configurations from the `space` to evaluate.

---

-   [__SMAC__](https://github.com/automl/SMAC3)

    SMAC is a collection of methods from [automl.org](https://www.automl.org) for hyperparameter optimization.
    Notably the library focuses on many Bayesian Optimization methods with highly
    configurable spaces.

---

-   [__Optuna__](https://optuna.org/)

    A hyperparameter optimization library which focuses on Tree-Parzan Estimators (TPE)
    for finding optimal configurations. There are some currentl limitations, such as
    heirarchical spaces but is widely used and popular.

---

-   [__NEPS__](https://automl.github.io/neps/0.8.3/)

    An optimization framework from [automl.org](https://www.automl.org) focusing
    on optimizing Neural Architectures.

    !!! info "Planned"

---

-   [__HEBO__](https://github.com/huawei-noah/HEBO)

    !!! info "Planned"

---

We note that [integrating in your own optimizer is fairly straightforward](#integrating-your-own-optimizer).
If there is an optimizer you would like integrated, please let us know!

## Running an Optimizer
Now that we have an optimizer that knows the `space` to search, we can begin to
actually [`ask()`][byop.optimization.Optimizer.ask] the optimizer for a next
[`Trial`][byop.optimization.Trial], run our function and return
a [`Trial.Report`][byop.optimization.Trial.Report].

First we need to modify our function we wish to optimize to actually accept
the `Trial` and return the `Report`.

=== "Running the optmizer"

    ```python hl_lines="4 5 6 7 8 9 10 19 20 21 22 24 25"
    from byop.optimization import RandomSearch
    from byop.pipeline import searchable

    def poly(trial):
        x = trial.config["x"]
        with trial.begin():  # (1)!
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)  # (2)!

        trial.fail()  # (3)!

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})

    space = s.space()
    random_search = RandomSearch(space=space, seed=42)

    results = []

    for _ in range(20):
        trial = random_search.ask()
        report = qaudratic(trial)
        random_search.tell(trial)

        cost = report.results["cost"]
        results.append(cost)
    ```

    1. Using the [`with trial.begin():`][byop.optimization.Trial.begin],
    you let us know where exactly your trial begins and we can handle
    all things related to exception handling and timing.
    2. If you can return a success, then do so with
    [`trial.success()`][byop.optimization.Trial.success].
    3. If you can't return a success, then do so with [`trial.fail()`][byop.optimization.Trial.fail].

=== "Typed"

    ```python hl_lines="4 5 6 7 8 9 10 19 20 21 22 24 25"
    from byop.optimization import RandomSearch, Trial
    from byop.pipeline import searchable

    def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:  # (4)!
        x = trial.config["x"]
        with trial.begin():  # (1)!
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)  # (2)!

        trial.fail()  # (3)!

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})

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

    1. Using the [`with trial.begin():`][byop.optimization.Trial.begin],
    you let us know where exactly your trial begins and we can handle
    all things related to exception handling and timing.
    2. If you can return a success, then do so with
    [`trial.success()`][byop.optimization.Trial.success]. This
    returns a [`Trial.SuccessReport`].
    3. If you can't return a success, then do so with [`trial.fail()`][byop.optimization.Trial.fail].
    This returns a [`trial.FailReport`][byop.optimization.Trial.FailReport].
    4. Here the inner type parameter `RSTrial` is the type of `trial.info` which
    contains the object returned by the ask of the wrapped `optimizer`. We'll
    see this in [integrating your own Optimizer](#integrating-your-own-optimizer).

### Running the Optimizer in a parallel fashion

Now that we've seen the basic optimization loop, it's time to parallelize it with
a [`Scheduler`][byop.scheduling.Scheduler] and the [`Trial.Task`][byop.optimization.Trial.Task].
We cover the [`Scheduler`][byop.scheduling.Scheduler] and [`Tasks`][byop.scheduling.Task]
in the [Tasks guide](./tasks.md) if you'd like to know more about how this works.

We first create a [`Scheduler`][byop.scheduling.Scheduler] to run with `#!python 1`
process and run it for `#!python 5` seconds.
Using the event system of AutoML-Toolkit,
we define what happens through _callbacks_, registering to certain events, such
as launch a single trial on `scheduler.on_start`, _tell_ the optimizer with `task.on_report`
and save results with `task.on_success`. 

=== "Creating a `Trial.Task`"

    ```python hl_lines="19 23 24 25 26 28 29 30 32 33 34 35 37 38 39 40 42"
    from byop.optimization import RandomSearch, Trial
    from byop.pipeline import searchable
    from byop.scheduling import Scheduler

    def poly(trial):
        x = trial.config["x"]
        with trial.begin():
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)

        trial.fail()

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})
    space = s.space()

    random_search = RandomSearch(space=space, seed=42)
    scheduler = Scheduler.with_processes(1)

    task = Trial.Task(poly)  # (5)!

    results = []

    @scheduler.on_start  # (1)!
    def launch_trial():
        trial = random_search.ask()
        task(trial)

    @task.on_report  # (2)!
    def tell_optimizer(report):
        random_search.tell(report)

    @task.on_report
    def launch_another_trial(_):
        trial = random_search.ask()
        task(trial)

    @task.on_success  # (3)!
    def save_result(report):
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
    [`Trial.Task`][byop.optimization.Trial]. There are other backends than
    processes, e.g. Clusters for which you should check out the [Task guide](./tasks.md).

=== "Typed"

    ```python hl_lines="19 23 24 25 26 28 29 30 32 33 34 35 37 38 39 40 42"
    from byop.optimization import RandomSearch, Trial, RSTrialInfo
    from byop.pipeline import searchable
    from byop.scheduling import Scheduler

    def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:
        x = trial.config["x"]
        with trial.begin():
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)

        trial.fail()

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})
    space = s.space()

    random_search = RandomSearch(space=space, seed=42)
    scheduler = Scheduler.with_processes(1)

    task = Trial.Task(poly)  # (5)!

    results: list[float] = []

    @scheduler.on_start  # (1)!
    def launch_trial() -> None:
        trial = random_search.ask()
        task(trial)

    @task.on_report  # (2)!
    def tell_optimizer(report: Trial.Report) -> None:
        random_search.tell(report)

    @task.on_report
    def launch_another_trial(_: Trial.Report) -> None:
        trial = random_search.ask()
        task(trial)

    @task.on_success  # (3)!
    def save_result(report: Trial.SuccessReport) -> None:
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
    [`Trial.Task`][byop.optimization.Trial]. There are other backends than
    processes, e.g. Clusters for which you should check out the [Task guide](./tasks.md).

!!! tip "Events of a `Trial.Task`"

    The `scheduler.on_start`, `task.on_report` and `task.on_success` methods
    define how our system runs. These are part of the event system of AutoML-Toolkit which
    are covered in more detail in the [Tasks guide](./tasks.md). Below we list some of the events
    related to a [`Trial.Task`][byop.optimization.Trial.Task].

    === "`on_success`"

        Any function decorated with this will be called when the wrapped function returns
        with [`trial.success()`][byop.optimization.Trial.success], passing on the
        [`Trial.SuccessReport`][byop.optimization.Trial.SuccessReport].

        === "Use"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_success
            def do_something_with_report(report):
                original_trial = report.trial
                original_config = report.trial.config
                time = report.time
                results = report.results
            ```

        === "Typed"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_success
            def do_something_with_report(report: Trial.SuccessReport) -> None:
                original_trial: Trial = report.trial
                original_config: dict[str, Any] = report.trial.config
                time: TimeInterval = report.time
                results: dict[str, Any] = report.results
            ```

    === "`on_failure`"

        Any function decorated with this will be called when the wrapped function returns
        with [`trial.fail()`][byop.optimization.Trial.fail], passing on the
        [`Trial.FailReport`][byop.optimization.Trial.FailReport].

        === "Use"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_failure
            def do_something_with_report(report):
                original_trial = report.trial
                original_config = report.trial.config
                time = report.time
                results = report.results
                exception = report.exception
            ```

        === "Typed"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_failure
            def do_something_with_report(report: Trial.FailReport) -> None:
                original_trial: Trial = report.trial
                original_config: dict[str, Any] = report.trial.config
                time: TimeInterval = report.time
                results: dict[str, Any] | None = report.results
                exception: BaseException | None = report.exception
            ```


    === "`on_crashed`"

        Any function decorated with this will be called when the wrapped function crashes
        outside of [`trial.begin()`][byop.optimization.Trial.begin]. In this case,
        we got nothing back from the called function exception an exception.
        We pass on this exception with a
        [`Trial.CrashReport`][byop.optimization.Trial.CrashReport].

        === "Use"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_crashed
            def do_something_with_report(report):
                original_trial = report.trial
                original_config = report.trial.config
                exception = report.exception
            ```

        === "Typed"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_crashed
            def do_something_with_report(report: Trial.CrashReport) -> None:
                original_trial: Trial = report.trial
                original_config: dict[str, Any] = report.trial.config
                exception: BaseException | None = report.exception
            ```

    === "`on_report`"

        Any function decorated with this will be called whenever any of the other
        events is triggered. It will pass on one of:

        * [SuccessReport][byop.optimization.Trial.SuccessReport]
        * [FailReport][byop.optimization.Trial.FailReport]
        * [CrashReport][byop.optimization.Trial.CrashReport]

        === "Use"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_report
            def do_something_with_report(report):
                original_trial = report.trial
                original_config = report.trial.config
            ```

        === "Typed"

            ```python
            from byop.optimization import Trial

            task = Trial.Task(...)

            @task.on_report
            def do_something_with_report(report: Trial.Report) -> None:
                original_trial: Trial = report.trial
                original_config: dict[str, Any] = report.trial.config
            ```


Now, to scale up, we trivially increase the number of initial trails launched with `scheduler.on_start`
and the number of processes in our `Scheduler`. That's it.

=== "Scaling Up"

    ```python hl_lines="18 19 25"
    from byop.optimization import RandomSearch, Trial
    from byop.pipeline import searchable
    from byop.scheduling import Scheduler

    def poly(trial):
        x = trial.config["x"]
        with trial.begin():
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)

        trial.fail()

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})
    space = s.space()

    random_search = RandomSearch(space=space, seed=42)

    n_workers = 4
    scheduler = Scheduler.with_processes(n_workers)

    task = Trial.Task(poly)

    results = []

    @scheduler.on_start(repeat=n_workers)
    def launch_trial():
        trial = random_search.ask()
        task(trial)

    @task.on_report
    def tell_optimizer(report):
        random_search.tell(report)

    @task.on_report
    def launch_another_trial(_):
        trial = random_search.ask()
        task(trial)

    @task.on_success
    def save_result(report):
        cost = report["cost"]
        results.append(cost)

    scheduler.run(timeout=5)
    ```

=== "Typed"

    ```python hl_lines="18 19 25"
    from byop.optimization import RandomSearch, Trial, RSTrialInfo
    from byop.pipeline import searchable
    from byop.scheduling import Scheduler

    def poly(trial: Trial[RSTrialInfo]) -> Trial.Report[RSTrialInfo]:
        x = trial.config["x"]
        with trial.begin():
            y = (x**2 + 4*x + 3) / x
            return trial.success(cost=y)

        trial.fail()

    s = Searchable("parameters", space={"x": (-10.0, 10.0)})
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

    @task.on_report
    def tell_optimizer(report: Trial.Report) -> None:
        random_search.tell(report)

    @task.on_report
    def launch_another_trial(_: Trial.Report) -> None:
        trial = random_search.ask()
        task(trial)

    @task.on_success
    def save_result(report: Trial.SuccessReport) -> None:
        cost = report["cost"]
        results.append(cost)

    scheduler.run(timeout=5)
    ```

That concludes the main portion of our `Optimization` guide. AutoML-Toolkit provides
a host of more useful options, such as:

* Setting constraints on your evaluation function, such as memory, wall time and cpu time, concurrency limits
and call limits. Please refer to the [Task guide](./tasks.md) for more information.
* Optimize over complex pipelines. Please refer to the [Pipeline guide](./pipelines.md) for more information.
* Using different parallelization strategies, such as [Dask](https://dask.org/), [Ray](https://ray.io/),
[Slurm](https://slurm.schedmd.com/), and [Apache Airflow](https://airflow.apache.org/).
* Use a whole host of more callbacks to control you system, check out the [Task guide](./tasks.md) for more information.
* Run the scheduler using `asyncio` to allow interactivity, run as a server or other more advanced use cases.


## Integrating your own Optimizer
To integrate you own optimizer, you'll need to implement the following interface:

```python
from byop.optimization import Trial, TrialInfo

class Optimizer(Protocol):
    def ask(self) -> Trial[TrialInfo]:
        ...

    def tell(self, report: Trial.Report[TrialInfo]) -> None:
        ...
```

The `ask` method should return a new `Trial` object, and the `tell` method should
update the optimizer with the result of the trial. The `TrialInfo` type is
whatever underlying information you want to store in the `Trial` object, under `trial.info`.

Each [`Trial`][byop.optimization.Trial] object has 3 required properties:

* `trial.name` - A unique name of the trial to identify it, as a `str`.
* `trial.config` - The configuration of the trial, as a [`Mapping[str, Any]`][typing.Mapping],
for example a simple `dict` with `str` keys and `Any` values.
* `trial.info` - Any underlying information of the trial that comes from your optimizer.

Here is a simplified example of wrapping [`SMAC`](https://automl.github.io/SMAC3/stable/).
The real implementation is more complex, but this should give you an idea of how to
implement your own optimizer.

```python hl_lines="8 13 24 22 47"
from smac.facade import AbstractFacade
from smac.runhistory import StatusType
from smac.runhistory import TrialInfo as SMACTrialInfo
from smac.runhistory import TrialValue as SMACTrialValue

from byop.optimization import Optimizer, Trial

class SMACOptimizer(Optimizer[SMACTrialInfo]):  # (1)!

    def __init__(self, *, facade: AbstractFacade) -> None:
        self.facade = facade

    def ask(self) -> Trial[SMACTrialInfo]:
        smac_trial_info = self.facade.ask()  # (2)!
        config = smac_trial_info.config
        budget = smac_trial_info.budget
        instance = smac_trial_info.instance
        seed = smac_trial_info.seed
        config_id = self.facade.runhistory.config_ids[config]

        unique_name = f"{config_id=}.{instance=}.{seed=}.{budget=}"  # (3)!
        return Trial(name=unique_name, config=config, info=smac_trial_info)  # (4)!

    def tell(self, report: Trial.Report[SMACTrialInfo]) -> None:  # (5)!
        if isinstance(report, (Trial.SuccessReport, Trial.FailReport)):
            duration = report.time.duration
            start = report.time.start,
            endtime = report.time.end,
            cost = report.results.get("cost", np.inf)
            status = (
                StatusType.SUCCESS
                if isinstance(report, Trial.SuccessReport)
                else StatusType.CRASHED
            )
            additional_info = report.results.get("additional_info", {})
        else:
            duration = 0
            start = 0
            end = 0
            reported_cost = np.inf
            additional_info = {}

        trial_value = SMACTrialValue( # (6)!
            time=duration,
            starttime=start,
            endtime=end,
            cost=cost,
            status=status,
            additional_info=additional_info,
        )
        self.facade.tell(trial_value)  # (7)!
```

1. We need to specify the type of `TrialInfo` that we'll be using. In this case, SMAC's `TrialInfo`
which they use internally.
2. We call `facade.ask()` to get a new `TrialInfo` object from SMAC.
3. We need to create a unique name for the trial, so we use the `config_id`, `instance`, `seed` and `budget`
to create a unique name.
4. We create a new `Trial` object, with the unique name, the configuration, and the `TrialInfo` object.
5. We need to implement the `tell` method, which is called when a trial is finished. We need to
report the results to the optimizer.
6. We create a `TrialValue` object, which is what SMAC uses internally to store the results of a trial.
7. We call `facade.tell` to report the results of the trial to SMAC.
