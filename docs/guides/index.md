The guides here serve as a well-structured introduction to the capabilities
of AutoML-Toolkit. Notably, we have three core concepts at the heart of
AutoML-Toolkit, with supporting types and auxiliary functionality to
enable these concepts.

These take the form of a **scheduling**, a **pipeline construction**
and **optimization**. By combining these concepts, we provide an extensive
framework from which to do AutoML research, utilize AutoML for you
task or build brand new AutoML systems.

---

-   **Scheduling**

    Dealing with multiple processes and simultaneous compute,
    can be both difficult in terms of understanding and utilization.
    Often a prototype script just doesn't work when you need to run
    larger experiments.

    We provide an **event-driven system** with a flexible **backend**,
    to help you write code that scales from just a few more cores on your machine
    to utilizing an entire cluster.

    This guide introduces `Task`s and the `Scheduler` in which they run, as well
    as `@events` which you can subscribe callbacks to. Define what should run, when
    it should run and simply define a callback to say what should happen when it's done.

    This framework allows you to write code that simply scales, with as little
    code change required as possible. Go from a single local process to an entire
    cluster with the same script and 5 lines of code.

    Checkout the [Scheduling guide!](./scheduling.md) for the full guide.
    We also cover some of these topics in brief detail in the reference pages.

    !!! tip "Notable Features"

        * A system that allows incremental and encapsulated feature addition.
        * An [`@event`](site:reference/scheduling/events.md) system with easy to use _callbacks_.
        * Place constraints and modify your [`Task`](site:reference/scheduling/task.md)
            with [`Plugins`](site:reference/scheduling/plugins.md)
        * Integrations for different [backends](site:reference/scheuling/executors.md) for where
        to run your tasks.
        * A wide set of events to plug into.
        * An easy way to extend the functionality provided with your own set of domain or task
            specific events.

---

-   **Pipelines**

    Optimizer require some _search space_ to optimize, yet provide no utility to actually
    define these search space. When scaling beyond a simple single model, these search space
    become harder to define, difficult to extend and are often disjoint from the actual pipeline
    creation. When you want to create search spaces that can have choices between models, parametrized
    pre-processing and a method to quickly change these setups, it can often feel tedious
    and error-prone

    By piecing together `Node`s of a pipeline, utilizing a set of different building blocks such
    as a `Component`, `Sequential`, `Choice`es and more, you can abstractly define your entire pipeline.
    Once you're done, we'll stitch together the entire `search_space()`, allow you to
    easily `configure()` it and finally `build()` it into a concrete object you can use,
    all in the same place.

    Checkout the [Pipeline guide!](./pipelines.md)
    We also cover some of these topics in brief detail in the reference pages.

    !!! tip "Notable Features"

        * An easy, declaritive pipeline structure, allowing for rapid addition, deletion and
          modification during experimentation.
        * A flexible pipeline capable of handling complex structures and subpipelines.
        * Mutliple component types to help you define your pipeline.
        * Exporting of pipelines into concrete implementations like an [sklearn.pipeline.Pipeline][]
          for use in your downstream tasks.
        * Extensible to add your own component types and `builder=`s to use.


---

-   **Optimization**

    An optimizer is the backbone behind many AutoML systems and the quickest way
    to improve the performance of your current pipelines. However optimizer's vary
    in terms of how they expect you to write code, they vary in how much control they
    take of your code and can be quite difficult to interact with other than
    their `run()` function.

    By setting a simple expectation on an `Optimizer`, e.g. that it should have
    an `ask()` and `tell()`, you are placed get back in terms of defining the loop,
    define what happens, when and you can store what you'd like to record and put it
    where you'd like to put it.

    By unifying their suggestions as a `Trial` and a convenient `Report` to hand back
    to them, you can switch between optimizers with minimal changes required. We have
    added a load of utility to the `Trial`'s, such that you can easily profile sections,
    add extra summary information, store artifacts and export DataFrames.

    Checkout the [Optimization guide](./optimization.md). We recommend reading the previous
    two guides to fully understand the possibilities with optimization.
    We also cover some of these topics in brief detail in the reference pages.

    !!! tip "Notable Features"

        * An assortment of different optimizers for you to swap in an out with relative ease
        through a unified interface.
        * A suite of utilities to help you record that data you want from your HPO experiments.
        * Full control of how you interact with it, allowing for easy warm-starting, complex
        swapping mechanisms or custom stopping criterion.
        * A simple interface to integrate in your own optimizer.

