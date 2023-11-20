Welcome to the AutoML-Toolkit framework docs.

!!! tip

    See the navigation links in the header or side-bars. Click the :octicons-three-bars-16: button (top left) on mobile.

For a quick-start, check out [examples](./examples/index.md) for copy-pastable
snippets to start from. For a more guided tour through what AutoML-Toolkit can offer, please check
out our [guides](./guides/index.md). If you've used AutoML-Toolkit before but need some refreshers, you can look
through our [reference pages](./reference/index.md) or the [API docs](./api/index.md).

## What is AutoML-Toolkit?

AutoML-Toolkit is a highly-flexible set of modules and components,
allowing you to define, search and build machine learning systems.



-   :material-language-python: **Python**

    Use the programming language that defines modern machine learning research.
    We use [mypy](https://mypy.readthedocs.io/en/stable/) internally and for external
    API so you can identify and fix errors before a single line of code runs.

---

-   :octicons-package-dependents-16: __Minimal Dependencies__

    AutoML-Toolkit was designed to not introduce dependencies on your code.
    We support some tool integrations but only if they are optionally installed!.

---

-   :material-connection: __Plug-and-play__

    We can't support all frameworks, and thankfully we don't have to. AutoML-Toolkit was
    designed to be plug-and-play. Integrate in your own
    [optimizers](./reference/optimization/optimizers.md),
    [search spaces](./reference/pipelines/spaces.md),
    [execution backends](./reference/scheduling/executors.md),
    [builders](./reference/pipelines/builders.md)
    and more.

    We've worked hard to make sure that how we integrate tools can be done for
    your own tools we don't cover.

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
    We provide a base [Task](./guides/scheduling.md) which you can extend with
    events and functionality specific to the tasks you care about.
