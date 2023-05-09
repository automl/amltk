The guides here serve as a well-structured introduction to the capabilities
of AutoML-Toolkit. Notably, we have three core concepts at the heart of
AutoML-Toolkit, with supporting types and auxiliary functionality to
enable these concepts.

These take the form of a [`Task`][byop.scheduling.Task], a [`Pipeline`][byop.pipeline.Pipeline]
and an [`Optimizer`][byop.optimization.Optimizer] which combines the two
to create the most flexible optimization framework we could imagine.

---

-   **Task**

    A `Task` is a function which we want to run _somewhere_, whether it be a local
    process, on some node of a cluster or out in the cloud. Equipped with an
    [`asyncio`][asyncio] **event-system** and a [`Scheduler`][byop.scheduling.Scheduler]
    to drive the gears of the system, we can provide a truly flexible and performant framework
    upon to which to build an AutoML system.

    !!! tip "Notable Features"

        * A system that allows incremental and encapsulated feature addition.
        * An event-driven system with easy to use _callbacks_.
        * Place constraints on your `Task`.
        * Integrations for different backends for where to run your tasks.
        * A wide set of events to plug into.
        * An easy to extend system to create your own specialized events and tasks.

    Checkout the [Task guide](./tasks.md)

---

-   **Pipeline**

    A [`Pipeline`][byop.pipeline.Pipeline] is a definition,
    defining what your **pipeline** will do and how
    it can be parametrized. By piecing together [`steps`][byop.pipeline.api.step],
    [`choices`][byop.pipeline.api.choice] and [`splits`][byop.pipeline.api.split], you can
    say how your pipeline should look and how it's parametrized. We'll take care
    of creating the search space to optimize over, configuring it and finally assembling
    it into something you can actually use.

    !!! tip "Notable Features"

        * An easy to edit pipeline structure, allowing for rapid addition, deletion and
          modification during experimentation.
        * A flexible pipeline capable of handling complex structures and subpipelines.
        * Easily attachable modules for things close to your pipeline but not a direct
          part of the main structure.
        * Exporting of pipelines into concrete implementations like an [sklearn.pipeline.Pipeline][]
          for use in your downstream tasks.

    Checkout the [Pipeline guide](./pipelines.md)

---

-   **Optimizer**

    An [`Optimizer`][byop.optimization.Optimizer] is the capstone of the preceding two
    fundamental systems. By leveraging an _"ask-and-tell"_ interface, we put you back
    in control of how your system interacts with the optimizer. You run what you want,
    wherever you want, telling the optimizer what you want and you storing what you want,
    wherever you want.
    This makes leveraging different optimizers easier than ever. By capturing the high-level
    core loop of black box optimization into a simple [`Trial`][byop.optimization.Trial] and
    a [`Report`][byop.optimization.Trial.Report], integrating your own optimizer is easy and
    provides the entire system that AutoML-Toolkit offers with little cost.

    !!! tip "Notable Features"

        * An assortment of different optimizers for you to swap in an out with relative ease
        through a unified interface.
        * Full control of how you interact with it, allowing for easy warm-starting, complex
        swapping mechanisms or custom stopping criterion.
        * A simple interface to integrate in your own optimizer.

    Checkout the [Optimization guide](./optimization.md). We recommend reading the previous
    two guides to fully understand the possibilities with optimization.
