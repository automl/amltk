AutoML-toolkit was built to support future development of AutoML systems and
a central part of an AutoML system is its Pipeline. The purpose of this
guide is to help you understand all the utility AutoML-toolkit can
provide to help you define your pipeline. We will do this by introducing concepts
from the ground up, rather than top down.
Please see [examples](./examples) if you would rather see copy-pastable examples.

---

## Introduction

At the core of a [`Pipeline`][byop.pipeline.Pipeline] definition
is the many [`Steps`][byop.pipeline.Step] it consists of.
By combining these together, you can define a _directed acyclic graph_ (DAG),
that repesents the flow of data through your [`Pipeline`][byop.pipeline.Pipeline].
Here is one such example which we will build up towards:

![Pipeline Overview Img](../images/pipeline-guide-overview.excalidraw.svg)

There are a few concrete flavours that each encode the different parts of
this DAG with which we can then search over.

* [`Component`][byop.pipeline.Component]: An step of a pipeline with
  an object attached and possibly some space attached to it.
* [`Choice`][byop.pipeline.Choice]: A step which represents the pipeline has a choice of which
  step(s) is next.
* [`Split`][byop.pipeline.Split]: A step which represents that the data flow
  through a pipeline will be split between the following steps. This is
  usually accompanied by some object that does the data splitting.
* [`Option`][byop.pipeline.Option] : A step which indicates the following
  step(s) are optionally included.

Once we have our pipeline definition, extracting a search space, configuring
it and building it into something useful can be done with the methods,
[`pipeline.space()`][byop.pipeline.Pipeline.space],
[`pipeline.sample()`][byop.pipeline.Pipeline.sample],
[`pipeline.configure()`][byop.pipeline.Pipeline.configure],
[`pipeline.build()`][byop.pipeline.Pipeline.build],

```python
from byop.Pipeline imort Pipeline

pipeline = Pipeline.create(...)

# Get the space for the pipeline and sample a concrete pipeline
space = pipeline.space()
config = pipeline.sample(space)

# Configure the pipeline
configured_pipeline = pipeline.config(config)

# Build the pipeline
built_pipeline = pipeline.build()
```

By the end of this guide you should be able to understand each of these
components, how to create them, modify it, and build your own
[`Pipeline`][byop.pipeline.Pipeline].

!!! tip

    We include some type hints such as `mystep: Component`
    throughout the example code but these types are optional and only
    there for reference.

!!! Note

    This guide requires both the ConfigSpace and sklearn integrations
    to be active with `pip install sklearn configspace`
    or `pip install amltk[sklearn, configspace]`

## First step
We'll start by creating a simple `Component`, showing how to give it
a search space, configure it and build it into something useful.


### Defining a Component
```python
from byop.pipeline import step, Component

mystep: Component = step("hello_step", object())  # (1)!
```

1. Using keywords, `#!python step(name="hello_step", item=object())`

While this particular component isn't very useful, we can highlight two important
facts, we can associate any `object` with a step and give it a `name`, in this
case `#!python "hello_step"`.

Lets create a step that will represent a Random Forest in our pipeline and set
some parameters for it.

```python
from byop.pipeline import step
from sklearn.ensemble import RandomForestclassifier

rf = step("rf", RandomForestClassifier, config={"n_estimators": 10})
```

Here the `config` is the parameters we want to call `RandomForestClassifier`
with whenever we want to turn our pipeline definition into something concrete.

If at any point we want to convert this step into the actual `RandomForestClassifier`
object, we can always call [`build()`][byop.pipeline.Component.build] on the step.

```python hl_lines="5"
from byop.pipeline import step
from sklearn.ensemble import RandomForestclassifier

rf = step("rf", RandomForestClassifier, config={"n_estimators": 10})
classifier = rf.build()  # RandomForestClassifier(n_estimators=10)

classifier.fit(...)
classifier.predict(...)
```

This by itself as you're probably noticed is a lot of work to just build a
random forest, and you're right. However the reason to wrap everything in a step
is for what comes next.


### Defining a search space

In any machine learning pipeline, we often wish to optimize some hyperparameters
of our model. For this we actually need to define some hyperparameters. For
this example, we will be using [`ConfigSpace`](../integrations/configspace.md) and
it's syntax for defining a search space, but check out our built-in
[integrations](../integrations) for more.

```python exec="on" source="tabbed-left" result="ansi"
from byop.pipeline import step
from sklearn.ensemble import RandomForestClassifier

rf = step(
    "rf",
    RandomForestClassifier,
    space={
        "n_estimators": (10, 100),
        "criterion": ["gini", "entropy", "log_loss"]
    },
    config={"max_depth": 5}
)
print(rf)  # markdown-exec: hide
```

Here we've told `amltk` that we have a component that `RandomForestClassifier`
has two hyperparameters we care about, some integer called `#!python "n_estimators"`
between `#!python (10, 100)` and some choice `#!python "criterion"` from
`#!python ["gini", "entropy", "log_loss"]`. We additionally specify that the
`#!python "max_depth"` should be `#!python 5` and is not part of the search space.
For the specifics of what these hyperparameters mean,
please [refer to the scikit-learn docs][sklearn.ensemble.RandomForestClassifier].

We can retrieve a space from this component by calling [`space()`] on it.

```python exec="on" source="tabbed-left" result="ansi" hl_lines="14"
from byop.pipeline import step
from sklearn.ensemble import RandomForestClassifier

rf = step(
    "rf",
    RandomForestClassifier,
    space={
        "n_estimators": (10, 100),
        "criterion": ["gini", "entropy", "log_loss"]
    },
    config={"max_depth": 5}
)

space = rf.space(seed=1)

print(space)  # markdown-exec: hide
```

To can create a simple configuration of our component or use `sample()`
on it to get a configuration that we can `configure()` with.

```python exec="on" source="tabbed-left" result="ansi" hl_lines="14"
from byop.pipeline import step
from sklearn.ensemble import RandomForestClassifier

rf = step(
    "rf",
    RandomForestClassifier,
    space={
        "n_estimators": (10, 100),
        "criterion": ["gini", "entropy", "log_loss"]
    },
    config={"max_depth": 5}
)

space = rf.space(seed=1)

manual_config = {"n_estimators": 5, "criterion": "gini"}
manual_rf = rf.configure(manual_config)
manual = manual_rf.build()

sampled_config = rf.sample(space)
sampled_rf = rf.configure(sampled_config)
sampled = sampled_rf.build()

print("manual")  # markdown-exec: hide
print(manual_rf)  # markdown-exec: hide
print(manual)  # markdown-exec: hide
print("\nsampled")   # markdown-exec: hide
print(sampled_rf)  # markdown-exec: hide
print(sampled) # markdown-exec: hide
```

```




, we need to be able to **pipe** them together.
For this we introduce a handy pipe operator `|`, inspired by the pipe operator from
shell scripting. This will return the first step in the chain but with anything after
that attached to it.
We can also [`append()`][byop.pipeline.Step.append] to existing chains.

```python hl_lines="3 8 9"
from byop.pipeline import step

chain = step("step_1", object()) | step("step_2", object())

assert chain.name == "step_1"  # (1)!
assert chain.nxt.name = "step_2"

new_step = step("step_3", object())
chain = chain.append(new_step)  # (2)!

assert chain.tail.name = "step_3"
```

1. The variable reference by `chain` is always the head of the chain.
2. Notice here we assign the result to a new variable. Operations on steps
are **not** done in-place and instead return a new variable.
