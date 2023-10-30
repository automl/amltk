Welcome to the AutoML-Toolkit framework docs.

!!! tip

    See the navigation links in the header or side-bars. Click the :octicons-three-bars-16: button (top left) on mobile.

Check out the bottom of this page for a [quick start](#quick-start),
or for a more thorough understanding of
all that AutoML-Toolkit has to offer, check out our [guides](guides/index.md).

You can also check out [examples](examples/index.md) for copy-pastable
snippets to start from.

## What is AutoML-Toolkit?

AutoML-Toolkit is a highly-flexible set of modules and components,
allowing you to define, search and build machine learning systems.



-   :material-language-python: **Python**

    Use the programming language that defines modern machine learning research.
    We use [mypy](https://mypy.readthedocs.io/en/stable/) internally and for external
    API so you can identifiy and fix errors before a single line of code runs.

---

-   :octicons-package-dependents-16: __Minimal Dependencies__

    AutoML-Toolkit was designed to not introduce dependencies on your code.
    We support some [integrations](reference/index.md) but only if they are optionally installed!.

---

-   :material-connection: __Plug-and-play__

    We can't support all frameworks, and thankfully we don't have to. AutoML-Toolkit was
    designed to be plug-and-play. Integrate in your own
    [optimizers](reference/index.md#optimizers),
    [search spaces](reference/index.md#search-spaces),
    [backends](reference/index.md#scheduler-executors),
    [builders](reference/index.md#pipeline-builders)
    and more. All of our [reference](reference/index.md) are built using this same API.

---

-   :material-tune-vertical-variant: __Event Driven__

    AutoML-Toolkit is event driven, meaning you write code that reacts to
    events as they happen. You can ignore, extend and create new events that
    have meaning to the systems you build.
    This enables tools built from AutoML-Toolkit to support greater forms
    of interaction, automation and deployment.

---

-   :material-directions-fork: __Task Agnostic__

    AutoML-Toolkit is task agnostic, meaning you can use it for any machine learning task.
    We provide a base [Task](guides/scheduling.md) which you can extend with
    events and functionality specific to the tasks you care about.

---

-   :octicons-people-16: __Community Driven__

    AutoML-Toolkit is a community driven project, and we want to hear from you. We
    are always looking for new contributors, so if you have an idea or want to
    contribute, please [get in touch](contributing.md).

---

## Quick Start
What you can use it for depends on what you want to do.

=== "Create Machine Learning Pipelines"

    We provide a __declarative__ way to define entire machine learning pipelines and any
    hyperparameters that go along with it. Rapidly experiment with different setups,
    get their search [`space()`][amltk.Pipeline.space], get concrete configurations with a quick
    [`configure()`][amltk.Pipeline.configure]
    and finally [`build()`][amltk.Pipeline.build] out a real
    [sklearn.pipeline.Pipeline][], [torch.nn.Sequential][] or
    your own custom pipeline objects.

    Here's a brief example of how you can use AutoML-Toolkit to define a pipeline,
    with its hyperparameters, sample from that space and build out a sklearn pipeline
    with minimal amounts of code. For a more in-depth look at pipelines and its features,
    check out the [Pipelines guide](./guides/pipelines.md) documentation.

    ```python
    from amltk.pipeline import Pipeline, step, split, choice

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    pipeline = Pipeline.create(
        choice(
            "scaler",  # (1)!
            step("standard", StandardScaler),
            step("minmax", MinMaxScaler))
        ),
        choice(
            "algorithm",
            step(
                "rf",
                RandomForestClassifier,
                space={"n_estimators": [10, 100] } # (2)!
            ),
            step(
                "svm",
                SVC,
                space={"C": (0.1, 10.0), "kernel": ["linear", "rbf"]},
                config={"kernel": "rbf"}  # (3)!
            ),
        ),
    )

    space = pipeline.space()  # (4)!
    config = pipeline.sample(space) # (6)!
    configured_pipeline = pipeline.configure(config)  # (7)!
    sklearn_pipeline = pipeline.build()  # (5)!
    ```

    1. Define choices between steps in your pipeline, `amltk` will figure out how to encode this choice into
    the search space.
    2. Decide what hyperparameters to search for for your steps.
    3. Want to quickly set something constant? Use the `config` argument to set a value and remove it from the space
     automatically.
    4. Parse out the search space for the pipeline, let `amltk` figure it out
      or choose your own [`parse(parser=...)`](reference/index.md)
    5. Let `amltk` figure out what kind of pipeline you want, but you can also
      specify your own [`build(builder=...)`](reference/index.md)
    6. Sample a configuration from the search space.
    7. Configure the pipeline with a configuration.

=== "Optimize Machine Learning Pipelines"

    AutoML-Toolkit integrates with a variety of optimization frameworks, allowing you to
    quickly optimize your machine learning pipelines with your favourite optimizer.
    We leave the optimization flow, the target function, when to stop and even what you want
    the tell the optimizer, completely up to you.

    We do however provide all the tools necessary to express exactly what you want
    to have happen.

    Below is a short showcase of the many ways you can define how you want to
    optimize and control the optimization process. For a more in-depth look at the full set
    of features, follow the [Optimization](./guides/optimization.md) documentation.

    ```python
    from amltk.pipeline import Pipeline
    from amltk.optimization import Trial
    from amltk.scheduling import Scheduler
    from amltk.smac import SMACOptimizer

    def evaluate(trial: Trial, pipeline: Pipeline) -> Trial.Report:
        model = pipeline.configure(trial.config).build()

        with trial.begin():  # (1)!
            # Train and evaluate the model

        if not trial.exception:
          return trial.success(cost=...)  # (2)!

        return trial.fail()

    my_pipeline = Pipeline.create(...)

    optimizer = SMACOptimizer.create(pipeline.space(), seed=42) # (4)!

    n_workers = 8
    scheduler = Scheduler.with_processes(n_workers)  # (3)!
    task = scheduler.task(evaluate)

    @scheduler.on_start(repeat=n_workers) # (6)!
    def start_optimizing():
        trial = optimizer.ask()
        task(trial=trial, pipeline=my_pipeline)  # (5)!

    @task.on_done
    def start_another_trial(_):
          trial = optimizer.ask()
          task(trial=trial, pipeline=my_pipeline)

    @task.on_result
    def tell_optimizer(report: Trial.Report):
        optimizer.tell(report)

    @task.on_result
    def store_result(report: Trial.Report):
        ...  # (8)!

    @task.on_exception
    @task.on_cancelled
    def stop_optimizing(exception):
        print(exception)
        scheduler.stop() # (9)!

    scheduler.run(timeout=60) # (10)!
    ```

    1. We take care of the busy work, just let us know when the trial starts.
    2. We automatically fill in the reports for the optimizer, just let us
      know the cost and any other additional info.
    3. Create a scheduler with your own custom backend. We provide a few out of the box,
    but you can also [integrate your own](site:guides/scheduling.md).
    4. Create an optimizer over your search space,
    we provide a few optimizers of the box, but you can also [integrate your own](site:guides/optimization.md#integrating-your-own-optimizer).
    5. Calling the task runs it in a worker, whether it be a process, cluster node, AWS or
      whatever backend you decide to use.
    6. Say _what_ happens and _when_, when the scheduler says it's started, this function
      gets called `n_workers` times.
    7. Inform the optimizer of the report ... if you want.
    8. We don't know what data you want and where, that's up to you.
    9. Stop the whole scheduler whenever you like under whatever conditions make sense to you.
    10. And let the system run!

    You can wrap this in a class, create more complicated control flows and even utilize
    some more of the functionality of a [`Task`][amltk.Task] to do
    much more. We don't tell you how the control flow should or where data goes, this gives
    you as much flexibility as you need to get your research done.


=== "Build Machine Learning Tools"

    AutoML-Toolkit is a set of tools that are for the purpose of building an AutoML system,
    it is not an AutoML system itself. With the variety of AutoML systems out there, we
    decided to build this framework as an event driven system. The cool part is, you can
    define your own events, your own tasks and how the scheduler should operate.

    !!! info "TODO"

        Come up with a nice example of defining your own task and events
