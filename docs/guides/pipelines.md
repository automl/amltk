# Pipelines Guide
AutoML-toolkit was built to support future development of AutoML systems and
a central part of an AutoML system is its Pipeline. The purpose of this
guide is to help you understand all the utility AutoML-toolkit can
provide to help you define your pipeline. We will do this by introducing concepts
from the ground up, rather than top down.
Please see [examples](site:examples/index.md) if you would rather see copy-pastable examples.

---

## Introduction

At the core of a [`Pipeline`][amltk.pipeline.Pipeline] definition
is the many [`Steps`][amltk.pipeline.Step] it consists of.
By combining these together, you can define a _directed acyclic graph_ (DAG),
that represents the structure of your [`Pipeline`][amltk.pipeline.Pipeline].
Here is one such example that we will build up towards.

```python exec="true" source="tabbed-right" html="True" title="Pipeline"
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from amltk import step, split, group, Pipeline

categorical_preprocessing = (
    step("categorical_imputer", SimpleImputer, config={"strategy": "constant", "fill_value": "missing"})
    | step("one_hot_encoding", OneHotEncoder, config={"drop": "first"})
)
numerical_preprocessing = step("numeric_imputer", SimpleImputer, space={"strategy": ["mean", "median"]})

pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        group("categoricals", categorical_preprocessing),
        group("numerics", numerical_preprocessing),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    ),
    step(
        "rf",
        RandomForestClassifier,
        space={"n_estimators": (10, 100), "criterion": ["gini", "entropy", "log_loss"]},
    ),
    name="My Classification Pipeline"
)
from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small", width=120)  # markdown-exec: hide
```

??? tip "`rich` printing"

    To get the same output locally (terminal or Notebook), you can either
    call `thing.__rich()__`, use `from rich import print; print(thing)`
    or in a Notebook, simply leave it as the last object of a cell.

Once we have our pipeline definition, extracting a search space, configuring
it and building it into something useful can be done with the methods.

* [`pipeline.space()`][amltk.pipeline.Pipeline.space],
    Get a useable search space out of the pipeline to pass to an optimizer.

* [`pipeline.sample()`][amltk.pipeline.Pipeline.sample],
    Sample a valid configuration from the pipeline.

* [`pipeline.configure(config=...)`][amltk.pipeline.Pipeline.configure],
    Configure a pipeline with a given config

* [`pipeline.build()`][amltk.pipeline.Pipeline.build],
    Build a configured pipeline into some useable object.

## Component
A `Pipeline` consists of building blocks which we can combine together
to create a DAG. We will start by introducing the `Component`, the common operations,
and then show how to combine them together.

A [`Component`][amltk.pipeline.Component] is a single atomic step in a pipeline. While
you can construct a component directly, it is recommended to use the
[`step()`][amltk.pipeline.api.step] function to create one.

### Definition
```python exec="true" source="material-block" html="true" session="Pipeline-Component"
from dataclasses import dataclass

from amltk import step

@dataclass
class MyModel:
    f: float
    i: int
    c: str

mystep = step(
    "mystep",
    MyModel,
    space={"f": (0.0, 1.0), "i": (0, 10), "c": ["red", "green", "blue"]},
)
from amltk._doc import doc_print; doc_print(print, mystep, output="html", fontsize="small")  # markdown-exec: hide
```

You can also use a **function** instead of a class if that is preffered.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
def myfunc(f: float, i: int, c: str) -> MyModel:
    if f < 0.5:
        c = "red"
    return MyModel(f=f, i=i, c=c)

step_with_function = step(
    "step_with_function",
    myfunc,
    space={"f": (0.0, 1.0), "i": (0, 10), "c": ["red", "green", "blue"]},
)
from amltk._doc import doc_print; doc_print(print, step_with_function, output="html", fontsize="small")  # markdown-exec: hide
```

### Sample
We now have a basic `Component` that parametrizes the class `MyModel`. What can be quite useful
is to now [`sample()`][amltk.pipeline.Step.sample] from it to get a valid configuration.

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
config = mystep.sample(seed=1)
print(config)
```

### Space
If interacting with an `Optimizer`, you'll often require some search space object to pass to it.
To extract a search space from a `Component`, we can call [`space()`][amltk.pipeline.Step.space].

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
space = mystep.space(seed=1)
print(space)
```

??? tip "What type of space is this?"

    Depending on the libraries you have installed and the values inside `space`, we will attempt
    to produce a valid search space for you. In this case, we have a `ConfigSpace` implementation
    installed and so we get a `ConfigSpace.ConfigurationSpace` object. If you wish to use a different
    space, you can always pass a specific [`step.space(parser=...)`][amltk.pipeline.Step.space]. Do
    note that not all spaces support all features.

    === "`ConfigSpace`"

        ```python exec="true" source="material-block" result="python" session="Pipeline-Component"
        from amltk.configspace import ConfigSpaceAdapter

        configspace_space = mystep.space(parser=ConfigSpaceAdapter)
        print(configspace_space)
        ```

    === "`Optuna`"

        ```python exec="true" source="material-block" result="python" session="Pipeline-Component"
        from amltk.optuna import OptunaSpaceAdapter

        optuna_space = mystep.space(parser=OptunaSpaceAdapter)
        print(optuna_space)
        ```

    You may also construct your own parser and use that if desired.

### Configure
Pretty straight forward but what do we do with this `config`? Well we can
[`configure(config=...)`][amltk.pipeline.Step.configure] the component with it.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
configured_step = mystep.configure(config)
from amltk._doc import doc_print; doc_print(print, configured_step, output="html", fontsize="small")  # markdown-exec: hide
```

You'll notice that each variable in the space has been set to some value. We could also manually
define a config and pass that in. You are **not** obliged to fully specify this either.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
manually_configured_step = mystep.configure({"f": 0.5, "i": 1})
from amltk._doc import doc_print; doc_print(print, manually_configured_step, output="html")  # markdown-exec: hide
```

!!! tip "Immutable methods!"

    One thing you may have noticed is that we assigned the result of `configure(...)` to a new
    variable. This is because we do not mutate the original `mystep` and instead return a copy
    with all of the `config` variables set.

### Build
The last important thing we can do with a `Component` is to [`build()`][amltk.pipeline.Component.build]
it. Thisa step is very straight-forward for a `Component` and it simply calls the `.item` with the
config we have set.

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
the_built_model = configured_step.build()

# Same as if we did `configured_step.item(**configured_step.config)`
print(the_built_model)
```

You may also pass additional items to `build()` which will overwrite any config values set.

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
the_built_model = configured_step.build(f=0.5, i=1)
print(the_built_model)
```

### Parameter Requests
Sometimes you may wish to explicitly specify some value should be added to the `.config` during
`configure()` which would be difficult to include in the `config`, for example the `random_state`
of an sklearn estimator. You can pass these extra parameters into `configure(params={...})`, which
do not require any namespace prefixing.

For this reason, we have the concept of a [`request()`][amltk.pipeline.request], allowing
you to specify that a certain parameter should be added to the config during `configure()`.

```python exec="true" hl_lines="14 17 18" source="material-block" html="true" session="Pipeline-Parameter-Request"
from dataclasses import dataclass

from amltk import step, request

@dataclass
class MyModel:
    f: float
    random_state: int

mystep = step(
    "mystep",
    MyModel,
    space={"f": (0.0, 1.0)},
    config={"random_state": request("seed", default=42)}
)

configured_step_with_seed = mystep.configure({"f": 0.5}, params={"seed": 1337})
configured_step_no_seed = mystep.configure({"f": 0.5})
from amltk._doc import doc_print; doc_print(print, configured_step_with_seed, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, configured_step_no_seed, output="html", fontsize="small")  # markdown-exec: hide
```

If you explicitly require a parameter to be set, you can pass `required=True` to the request.

```python exec="true" source="material-block" result="python" session="Pipeline-Parameter-Request"
mystep = step(
    "mystep",
    MyModel,
    space={"f": (0.0, 1.0)},
    config={"random_state": request("seed", required=True)}
)

mystep.configure({"f": 0.5}, params={"seed": 5})  # All good

try:
    mystep.configure({"f": 0.5})  # Missing required parameter
except ValueError as e:
    print(e)
```

### Config Transform
Some search space and optimizers may have limitations in terms of the kinds of parameters they
can support, one notable example is tuple parameters. To get around this, we can pass
a `config_transform` to `step` which will transform the config before it is passed to the
`.item` during `build()`.

```python exec="true" hl_lines="9-13 19" source="material-block" html="true"
from dataclasses import dataclass

from amltk import step

@dataclass
class MyModel:
    dimensions: tuple[int, int]

def config_transform(config: dict, _) -> dict:
    dim1 = config.pop("dim1")
    dim2 = config.pop("dim2")
    config["dimensions"] = (dim1, dim2)
    return config

mystep = step(
    "mystep",
    MyModel,
    space={"dim1": (1, 10), "dim2": (1, 10)},
    config_transform=config_transform,
)

configured_step = mystep.configure({"dim1": 5, "dim2": 5})
from amltk._doc import doc_print; doc_print(print, configured_step, output="html", fontsize="small")  # markdown-exec: hide
```

Lastly, there may be times where you may have some additional context which you may only
know at configuration time, you may pass this to `configure(..., transform_context=...)`
which will be forwarded as the second argument to your `.config_transform`.

## Pipelines
A single step might be enough for some basic definitions but generally we need to combine multiple
steps. AutoML-Toolkit is designed for large and more complex structures which can be made from
simple atomic steps.

### Joining Steps
We'll begin by creating two components that wrap scikit-learn estimators.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from amltk import step

imputer_step = step("imputer", SimpleImputer, space={"strategy": ["median", "mean"]})
rf_step = step("random_forest", RandomForestClassifier, space={"n_estimators": (10, 100)})

from amltk._doc import doc_print; doc_print(print, imputer_step, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, rf_step, output="html", fontsize="small")  # markdown-exec: hide
```

!!! tip "Modifying Display Output"

    By default, `amltk` will show full function signatures, including a link to their documentation
    if available.

    You can control these by setting some global `amltk` options.

    ```python
    from amltk import options

    options["rich_signatures"] = False
    ```

    You can find the [available options here][amltk.options.AMLTKOptions].

To join these two steps together, we can either use the infix notation using `|` (reminiscent of a bash pipe)
or directly call [`append(nxt)`][amltk.pipeline.Step.append] on the first step.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
joined_steps = imputer_step | rf_step
from amltk._doc import doc_print; doc_print(print, joined_steps, output="html", fontsize="small")  # markdown-exec: hide
```

We should point out two key things here:

* You are always returned the _head_ of the steps, i.e. the first step in the list
* You can see the `rf_step` is now attached to the `imputer_step` as its `nxt` attribute.

However viewing only one step at a time is not so useful. We can get a [`Pipeline`][amltk.pipeline.Pipeline]
out of these steps quite easily which will display a lot more nicely and allow you to perform operations

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
pipeline = joined_steps.as_pipeline(name="My Pipeline")
from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

??? tip "Using `Pipeline.create(...)` instead"

    You can also use [`Pipeline.create(...)`][amltk.pipeline.Pipeline.create] to create a pipeline
    from a set of steps.

    ```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
    from amltk import Pipeline

    pipeline2 = Pipeline.create(joined_steps, name="My Pipeline")
    from amltk._doc import doc_print; doc_print(print, pipeline2, output="html", fontsize="small")  # markdown-exec: hide
    ```

### Pipeline Usage

You can perform much of the same operations as we did for the individual step but now taking into account
everything in the pipeline.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
space = pipeline.space()
config = pipeline.sample(seed=1337)
configured_pipeline = pipeline.configure(config)
from amltk._doc import doc_print; doc_print(print, space, output="html")  # markdown-exec: hide
doc_print(print, config, output="html")  # markdown-exec: hide
doc_print(print, configured_pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

Pipelines also support a number of other operations such as traversal with [`iter()`][amltk.pipeline.Pipeline.iter],
[`traverse()`][amltk.pipeline.Pipeline.traverse] and [`walk()`][amltk.pipeline.Pipeline.walk],
search with [`find()`][amltk.pipeline.Pipeline.find], modification with [`remove()`][amltk.pipeline.Pipeline.remove],
[`apply()`][amltk.pipeline.Pipeline.apply] and [`replace()`][amltk.pipeline.Pipeline.replace].

### Pipeline Building
Perhaps the most significant difference when working with a `Pipeline` is what should something
like [`build()`][amltk.pipeline.Pipeline.build] do? Well, there are perhaps multiple steps and perhaps
even nested `choice` and `split` components which we will introduce later.

The answer depends on what is contained within your steps. For this example, using sklearn, we can
directly return an sklearn [`Pipeline`][sklearn.pipeline.Pipeline] object. This is auto detected
based on the `.item` contained in each step.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
from sklearn.pipeline import Pipeline as SklearnPipeline

built_pipeline = configured_pipeline.build()
assert isinstance(built_pipeline, SklearnPipeline)
print(built_pipeline._repr_html_())  # markdown-exec: hide
```

We currently support the following builders which are auto detected based on the `.item` contained:

=== "Scikit-Learn"

    Using [`sklearn_pipeline()`][amltk.sklearn.sklearn_pipeline] will builds
    an [`SklearnPipeline`][sklearn.pipeline.Pipeline] from the steps. The
    possible pipelines allowed follow the rules of an sklearn pipeline, i.e.
    only the final step can be an estimator and everything else before it must
    be a transformer.

    ```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
    from amltk.sklearn import sklearn_pipeline

    built_pipeline = configured_pipeline.build(builder=sklearn_pipeline)
    print(built_pipeline._repr_html_())
    ```

    If using something like `imblearn` components, you will need to have an
    `imblearn.pipeline.Pipeline` as output type. We can pass this directly to
    the builder.

    ```python
    from amltk.sklearn import sklearn_pipeline
    from imblearn.pipeline import Pipeline as ImblearnPipeline

    built_pipeline = configured_pipeline.build(
        builder=sklearn_pipeline,
        pipeline_type=ImblearnPipeline
    )
    ```

=== "`pytorch_builder()`"

    !!! todo "TODO"

        This is currently in progress and will be available soon. Please
        feel free to reach out to help

=== "Custom Builder"

    You can also provide your own `builder=` function which has a very basic premise that
    you must be able to parse the pipeline and return _something_. Here is a basic example
    which will just return a dict with the step names as keys and the values as the built
    components.

    ```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Steps"
    def mybuilder(pipeline: Pipeline, **kwargs) -> dict:
        return {step.name: step.build() for step in pipeline.traverse()}

    components = configured_pipeline.build(builder=mybuilder)
    ```

## Building blocks
We saw the basic building block of a `Component` but AutoML-Toolkit also provides support
for some other kinds of building blocks. These building blocks can be attached and appended
just like a `Component` can and allow for much more complex pipeline structures.

### Choice
A [`Choice`][amltk.pipeline.Choice] is a way to define a choice between multiple
components. This is useful when you want to search over multiple algorithms, which
may each have their own hyperparameters.

The preferred way to create a `Choice` is to use the [`choice(...)`][amltk.pipeline.choice]
function.

We'll start again by creating two steps:

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
from dataclasses import dataclass

from amltk import step

@dataclass
class ModelA:
    i: int

@dataclass
class ModelB:
    c: str

model_a = step("model_a", ModelA, space={"i": (0, 100)})
model_b = step("model_b", ModelB, space={"c": ["red", "blue"]})
from amltk._doc import doc_print; doc_print(print, model_a, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, model_b, output="html", fontsize="small")  # markdown-exec: hide
```

Now combining them into a choice is rather straight forward:

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
from amltk import choice

model_choice = choice("model", model_a, model_b)
from amltk._doc import doc_print; doc_print(print, model_choice, output="html", fontsize="small")  # markdown-exec: hide
```

Just as we did with a `Component`, we can also get a `space()` from the choice. If the space
parser supports conditionals from a space, it will even add conditionals to the space to
account for the choice and that some hyperparameters are only active depending on if the
model is chosen.

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
space = model_choice.space()
from amltk._doc import doc_print; doc_print(print, space, output="html")  # markdown-exec: hide
```

When we `configure()` a choice, we will collapse it down to a single component. This is
done according to what is set in the config.

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
config = model_choice.sample(seed=1)
configured_model = model_choice.configure(config)
from amltk._doc import doc_print; doc_print(print, configured_model, output="html")  # markdown-exec: hide
```

### Group
The purpose of a [`Group`][amltk.pipeline.Group] is to _"draw a box"_ around a certain
subsection of a pipeline. This essentially acts as a namespacing mechanism for the
config and space of the steps contained within it. This can be useful
when you need to refer to a `Choice` in part of a `Pipeline`, where when configured,
this `Choice` will disappear and be replaced by a single component.

To illustrate this, let's revisit what happens when we `configure()` a choice. First we'll
build a small pipeline.

```python exec="true" source="material-block" html="true" session="Pipeline-Group"
from amltk import step, choice, group

model_a = step("model_a", object)
model_b = step("model_b", object)

preprocessing = step("preprocessing", object)
model_choice = choice("classifier_choice", model_a, model_b)

pipeline = (preprocessing | model_choice).as_pipeline(name="My Pipeline")

from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

Now let's sample a config from the pipeline space and `configure()` it to see what we get out.


```python exec="true" source="material-block" html="true" session="Pipeline-Group"
space = pipeline.space()
config = pipeline.sample(seed=1)
configured_pipeline = pipeline.configure(config)
doc_print(print, space, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, config, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, configured_pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

We can't know ahead of time whether we need to refer to `#!python "model_a"`
or `#!python "model_b"` and we can no longer refer to `#!python "classifier_choice"` as this
has been configured away.

To circumvent this, we can use a [`Group`][amltk.pipeline.Group] to wrap the choice, with
the preferred way to create one being [`group(...)`][amltk.pipeline.group].

```python exec="true" source="material-block" html="true" session="Pipeline-Group"
from amltk import step, choice, group

model_a = step("model_a", object)
model_b = step("model_b", object)

preprocessing = step("preprocessing", object)
classifier_group = group(
    "classifier",
    choice("classifier_choice", model_a, model_b)
)

pipeline = (preprocessing | classifier_group).as_pipeline(name="My Pipeline")
from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

Now let's configure it:
```python exec="true" source="material-block" html="true" session="Pipeline-Group"
config = pipeline.sample(seed=1)
configured_pipeline = pipeline.configure(config)
doc_print(print, config, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, configured_pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

If we need to access the chosen classifier, we can do so in a straightforward manner:
```python exec="true" source="material-block" html="true" session="Pipeline-Group"
chosen_classifier = configured_pipeline.find("classifier").first()
doc_print(print, chosen_classifier, output="html", fontsize="small")  # markdown-exec: hide
```


### Split
A [`Split`][amltk.pipeline.Split] is a way to signify a split in the dataflow of a pipeline,
with the preferred way to create one being [`split(...)`][amltk.pipeline.split]. This `Split`
by itself will not do anything but it informs the builder about what to do. Each builder
will have if it's own specific strategy for dealing with one.

Before we go ahead with a full scikit-learn example and build it, we'll start with
an abstract representation of a `Split`.

```python exec="true" source="material-block" html="True" session="Pipeline-Split1"
from amltk import step, split, group, Pipeline

preprocesser = split(
    "preprocesser",
    step("cat_imputer", object) | step("cat_encoder", object),
    step("num_imputer", object),
    config={"cat_imputer": object, "num_imputer": object}
)
from amltk._doc import doc_print; doc_print(print, preprocesser, output="html", fontsize="small", width=120)  # markdown-exec: hide
```

You'll notice that if we have any hope to configure this `Split` which normally requires
mentioning each of it's paths, we can only reference the first step of the path, in this
case `#!python "cat_imputer"` and `#!python "num_imputer"`. In the case of the first step
being a `Choice`, we may not even have a name we can refer to!

We fix this situation by giving each split path its own name. We can either do this manually
with a `Group` or we can simply pass a `dict` of paths to a sequence of steps.

```python exec="true" source="material-block" html="True" session="Pipeline-Split2"
from amltk import step, split, group, Pipeline

preprocesser = split(
    "preprocesser",
    {
        "categories": step("cat_imputer", object) | step("cat_encoder", object),
        "numericals": step("num_imputer", object),
    },
    config={"categories": object, "numericals": object}
)
from amltk._doc import doc_print; doc_print(print, preprocesser, output="html", fontsize="small", width=120)  # markdown-exec: hide
```

This construction will use a `Group` around each of the paths, which will allow us to refer
to the different paths, regardless of what happens to the path.

Now this time we will use a scikit-learn example, as an example.

```python exec="true" source="material-block" html="True" session="Pipeline-Split3"
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from amltk import step, split, group, Pipeline

# We'll impute categorical features and then OneHotEncode them
category_pipeline = step("categorical_imputer", SimpleImputer) | step("one_hot_encoding", OneHotEncoder)

# We just impute numerical features
numerical_pipeline = step("numeric_imputer", SimpleImputer, config={"strategy": "median"})

feature_preprocessing = split(
    "feature_preprocessing",
    group("categoricals", category_pipeline),
    group("numerics", numerical_pipeline),
    item=ColumnTransformer,
    config={
        # Here we specify which columns should be passed to which group
        "categoricals": make_column_selector(dtype_include=object),
        "numerics": make_column_selector(dtype_include=np.number),
    },
)
from amltk._doc import doc_print; doc_print(print, feature_preprocessing, output="html", fontsize="small", width=120)  # markdown-exec: hide
```

Our last step is just to convert this into a `Pipeline` and `build()` it. First,
to convert it into a pipeline with a classifier at the end.

```python exec="true" source="material-block" html="True" session="Pipeline-Split3"
from sklearn.ensemble import RandomForestClassifier

classifier = step("random_forest", RandomForestClassifier)
pipeline = (feature_preprocessing | classifier).as_pipeline(name="Classification Pipeline")
from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small", width=120)  # markdown-exec: hide
```

And finally, to build it:
```python exec="true" source="material-block" html="True" session="Pipeline-Split3"
from sklearn.pipeline import Pipeline as SklearnPipeline

sklearn_pipeline = pipeline.build()
assert isinstance(sklearn_pipeline, SklearnPipeline)
print(sklearn_pipeline._repr_html_())  # markdown-exec: hide
```

### Option

!!! todo "TODO"

    Please feel free to provide a contribution!

## Modules
A pipeline is often not sufficient to represent everything surrounding the pipeline
that you'd wish to associate with it. For that reason we introduce the concept
of _module_.
These are components or pipelines that you [`attach()`][amltk.pipeline.Pipeline.attach]
to your main pipeline, but are not directly part of the dataflow.

For example, we can create a simple [`searchable()`][amltk.pipeline.api.searchable]
which we `attach()` to our pipeline.
This will be included in the `space()` that it outputed from the `Pipeline.

```python exec="true" source="material-block" html="True" session="Pipeline-Modules"
from amltk.pipeline import step, searchable

# Some extra things we want to include in the search space of the pipeline
params_a = searchable("params_a", space={"a": (1, 10), "b": ["apple", "frog"]})
params_b = searchable("params_b", space={"c": (1.5, 1.8)})

# Create a basic pipeline of two steps
pipeline = (step("step1", object) | step("step2", object)).as_pipeline()
pipeline = pipeline.attach(modules=[params_a, params_b])

space = pipeline.space()
from amltk._doc import doc_print; doc_print(print, pipeline, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, space, output="html", fontsize="small")  # markdown-exec: hide
```

These will also be included in any configurations `sample()`'ed and will be configured with
`configure()`.

```python exec="true" source="material-block" html="True" session="Pipeline-Modules"
config = pipeline.sample()
pipeline = pipeline.configure(config)

doc_print(print, config, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, pipeline, output="html", fontsize="small")  # markdown-exec: hide
```

Lastly, we can access the config directly through the pipelines `.modules`

```python exec="true" source="material-block" html="True" session="Pipeline-Modules"
module_config = pipeline.modules["params_a"].config
doc_print(print, module_config, output="html", fontsize="small")  # markdown-exec: hide
```
