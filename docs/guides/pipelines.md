# Pipelines Guide
AutoML-toolkit was built to support future development of AutoML systems and
a central part of an AutoML system is its pipeline. The purpose of this
guide is to help you understand all the utility AutoML-toolkit can
provide to help you define your pipeline. We will do this by introducing concepts
from the ground up, rather than top down.
Please see [the reference](../reference/pipelines/pipeline.md)
if you just want to quickly look something up.

---

## Introduction
The kinds of pipelines that exist in an AutoML system come in many different
forms. For example, one might be an [sklearn.pipeline.Pipeline][], other's
might be some deep-learning pipeline while some might even stand for some
real life machinery process and the settings of these machines.

To accomodate this, what AutoML-Toolkit provides is an **abstract** representation
of a pipeline, to help you define its search space and also to build concrete
objects in code if possible (see [builders](../reference/pipelines/builders.md).

We categorize this into 4 steps:

1. Parametrize your pipeline using the various [components](../reference/pipelines/pipeline.md),
    including the kinds of items in the pipeline, the search spaces and any additional configuration.
    Each of the various types of components give a syntactic meaning when performing the next steps.

2. [`pipeline.search_space(parser=...)`][amltk.pipeline.Node.search_space],
    Get a useable search space out of the pipeline. This can then be passed to an
    [`Optimizer`](../reference/optimization/optimizers.md).

3. [`pipeline.configure(config=...)`][amltk.pipeline.Node.configure],
    Configure your pipeline, either manually or using a configuration suggested by
    an optimizers.

4. [`pipeline.build(builder=)`][amltk.pipeline.Node.build],
    Build your configured pipeline definition into something useable, i.e.
    an [`sklearn.pipeline.Pipeline`][sklearn.pipeline.Pipeline] or a
    `torch.nn.Module` (_todo_).

At the core of these definitions is the many [`Nodes`][amltk.pipeline.node.Node]
it consists of. By combining these together, you can define a _directed acyclic graph_ (DAG),
that represents the structure of your pipeline.
Here is one such sklearn example that we will build up towards.

```python exec="true" source="tabbed-right" html="True" title="Pipeline"
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from amltk.pipeline import Component, Split, Sequential

feature_preprocessing = Split(
    {
        "categoricals": [SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(drop="first")],
        "numerics": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
    },
    config={
        "categoricals": make_column_selector(dtype_include=object),
        "numerics": make_column_selector(dtype_include=np.number),
    },
    name="preprocessing",
)

pipeline = Sequential(
    feature_preprocessing,
    Component(RandomForestClassifier, space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]}),
    name="Classy Pipeline",
)
from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
```

??? tip "`rich` printing"

    To get the same output locally (terminal or Notebook), you can either
    call `thing.__rich()__`, use `from rich import print; print(thing)`
    or in a Notebook, simply leave it as the last object of a cell.

Once we have our pipeline definition, extracting a search space, configuring
it and building it into something useful can be done with the methods.

!!! tip "Guide Requirements"

    For this guide, we will be using `ConfigSpace` and `scikit-learn`, you can
    install them manually or as so:

    ```bash
    pip install "amltk[sklearn, configspace]"
    ```

## Component
A pipeline consists of building blocks which we can combine together
to create a DAG. We will start by introducing the `Component`, the common operations,
and then show how to combine them together.

A [`Component`][amltk.pipeline.Component] is the most common kind of node a pipeline.
Like all parts of the pipeline, they subclass [`Node`][amltk.pipeline.Node] but a
`Component` signifies this is some concrete object, with a possible
[`.space`][amltk.pipeline.Node.space] and [`.config`][amltk.pipeline.Node.config].


### Definition

??? tip inline end "Naming Nodes"

    By default, a `Component` (or any `Node` for that matter), will use the function/classname
    for the [`.name`][amltk.pipeline.Node.name] of the `Node`. You can explicitly pass
    a `name=` **as a keyword argument** when constructing these.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
from dataclasses import dataclass

from amltk.pipeline import Component

@dataclass
class MyModel:
    f: float
    i: int
    c: str

my_component = Component(
    MyModel,
    space={"f": (0.0, 1.0), "i": (0, 10), "c": ["red", "green", "blue"]},
)
from amltk._doc import doc_print; doc_print(print, my_component, output="html", fontsize="small")  # markdown-exec: hide
```

You can also use a **function** instead of a class if that is preferred.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
def myfunc(f: float, i: int, c: str) -> MyModel:
    if f < 0.5:
        c = "red"
    return MyModel(f=f, i=i, c=c)

component_with_function = Component(
    myfunc,
    space={"f": (0.0, 1.0), "i": (0, 10), "c": ["red", "green", "blue"]},
)
from amltk._doc import doc_print; doc_print(print, component_with_function, output="html", fontsize="small")  # markdown-exec: hide
```

### Search Space
If interacting with an [`Optimizer`](../reference/optimization/optimizers.md), you'll often require some
search space object to pass to it.
To extract a search space from a `Component`, we can call [`search_space(parser=)`][amltk.pipeline.Node.search_space],
passing in the kind of search space you'd like to get out of it.

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
space = my_component.search_space("configspace")
print(space)
```

!!! tip inline end "Available Search Spaces"

    Please see the [spaces reference](../reference/pipelines/spaces.md)

Depending on what you pass as the `parser=` to `search_space(parser=...)`, we'll attempt
to give you a valid search space. In this case, we specified `#!python "configspace"` and 
so we get a `ConfigSpace` implementation.

You may also define your own `parser=` and use that if desired.


### Configure
Pretty straight forward but what do we do with this `config`? Well we can
[`configure(config=...)`][amltk.pipeline.Node.configure] the component with it.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
config = space.sample_configuration()
configured_component = my_component.configure(config)
from amltk._doc import doc_print; doc_print(print, configured_component)  # markdown-exec: hide
```

You'll notice that each variable in the space has been set to some value. We could also manually
define a config and pass that in. You are **not** obliged to fully specify this either.

```python exec="true" source="material-block" html="true" session="Pipeline-Component"
manually_configured_component = my_component.configure({"f": 0.5, "i": 1})
from amltk._doc import doc_print; doc_print(print, manually_configured_component, output="html")  # markdown-exec: hide
```

!!! tip "Immutable methods!"

    One thing you may have noticed is that we assigned the result of `configure(config=...)` to a new
    variable. This is because we do not mutate the original `my_component` and instead return a copy
    with all of the `config` variables set.

### Build
To build the individual item of a `Component` we can use [`build_item()`][amltk.pipeline.Component.build_item]
and it simply calls the `.item` with the config we have set.

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
the_built_model = configured_component.build_item()

# Same as if we did `configured_component.item(**configured_component.config)`
print(the_built_model)
```

However, as we'll see later, we often have multiple steps of a pipeline joined together and so
we need some way to get a full object out of it that takes into account all of these items
joined together. We can do this with [`build(builder=...)`][amltk.pipeline.Node.build].

```python exec="true" source="material-block" result="python" session="Pipeline-Component"
the_built_model = configured_component.build(builder="sklearn")
print(the_built_model)
```

For a look at the available arguments to pass to `builder=`, see the
[builder reference](../reference/pipelines/builders.md)

### Fixed

Sometimes we just have some part of the pipeline with no search space and
no configuration required, i.e. just some prebuilt thing. We can
use the [`Fixed`][amltk.pipeline.Fixed] node type to signify this.

```python exec="true" source="material-block" result="python"
from amltk.pipeline import Fixed
from sklearn.ensemble import RandomForestClassifier

frozen_rf = Fixed(RandomForestClassifier(n_estimators=5))
from amltk._doc import doc_print; doc_print(print, frozen_rf)  # markdown-exec: hide
```

### Parameter Requests
Sometimes you may wish to explicitly specify some value should be added to the `.config` during
`configure()` which would be difficult to include in the `config` directly, for example the `random_state`
of an sklearn estimator. You can pass these extra parameters into `configure(params={...})`, which
do not require any namespace prefixing.

For this reason, we introduce the concept of a [`request()`][amltk.pipeline.request], allowing
you to specify that a certain parameter should be added to the config during `configure()`.

```python exec="true" source="material-block" html="true" session="Pipeline-Parameter-Request"
from dataclasses import dataclass

from amltk import Component, request

@dataclass
class MyModel:
    f: float
    random_state: int

my_component = Component(
    MyModel,
    space={"f": (0.0, 1.0)},
    config={"random_state": request("seed", default=42)}
)

# Without passing the params
configured_component_no_seed = my_component.configure({"f": 0.5})

# With passing the params
configured_component_with_seed = my_component.configure({"f": 0.5}, params={"seed": 1337})
from amltk._doc import doc_print; doc_print(print, configured_component_no_seed)  # markdown-exec: hide
doc_print(print, configured_component_with_seed)  # markdown-exec: hide
```

If you explicitly require a parameter to be set, just do not set a `default=`.

```python exec="true" source="material-block" result="python" session="Pipeline-Parameter-Request"
my_component = Component(
    MyModel,
    space={"f": (0.0, 1.0)},
    config={"random_state": request("seed")}
)

my_component.configure({"f": 0.5}, params={"seed": 5})  # All good

try:
    my_component.configure({"f": 0.5})  # Missing required parameter
except ValueError as e:
    print(e)
```

### Config Transform
Some search space and optimizers may have limitations in terms of the kinds of parameters they
can support, one notable example is **tuple** parameters. To get around this, we can pass
a `config_transform=` to `component` which will transform the config before it is passed to the
`.item` during `build()`.

```python exec="true" hl_lines="9-13 19" source="material-block" html="true"
from dataclasses import dataclass

from amltk import Component

@dataclass
class MyModel:
    dimensions: tuple[int, int]

def config_transform(config: dict, _) -> dict:
    """Convert "dim1" and "dim2" into a tuple."""
    dim1 = config.pop("dim1")
    dim2 = config.pop("dim2")
    config["dimensions"] = (dim1, dim2)
    return config

my_component = Component(
    MyModel,
    space={"dim1": (1, 10), "dim2": (1, 10)},
    config_transform=config_transform,
)

configured_component = my_component.configure({"dim1": 5, "dim2": 5})
from amltk._doc import doc_print; doc_print(print, configured_component, fontsize="small")  # markdown-exec: hide
```

!!! tip inline end "Transform Context"

    There may be times where you may have some additional context which you may only
    know at configuration time, you may pass this to `configure(..., transform_context=...)`
    which will be forwarded as the second argument to your `.config_transform`.

## Sequential
A single component might be enough for some basic definitions but generally we need to combine multiple
components together. AutoML-Toolkit is designed for large and more complex structures which can be
made from simple atomic [`Node`][amltk.pipeline.Node]s.

### Chaining Together Nodes
We'll begin by creating two components that wrap scikit-learn estimators.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Nodes"
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from amltk.pipeline import Component

imputer = Component(SimpleImputer, space={"strategy": ["median", "mean"]})
rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})

from amltk._doc import doc_print; doc_print(print, imputer)  # markdown-exec: hide
doc_print(print, rf)  # markdown-exec: hide
```

!!! info inline end "Modifying Display Output"

    By default, `amltk` will show full function signatures, including a link to their documentation
    if available.

    You can control these by setting some global `amltk` options.

    ```python
    from amltk import options

    options["rich_signatures"] = False
    ```

    You can find the [available options here][amltk.options.AMLTKOptions].


```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Nodes"
from amltk.pipeline import Sequential
pipeline = Sequential(imputer, rf, name="My Pipeline")
from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
```

!!! info inline end "Infix `>>`"

    To join these two components together, we can either use the infix notation using `>>`,
    or passing them directly to a [`Sequential`][amltk.pipeline.Sequential]. However
    a random name will be given.

    ```python
    joined_components = imputer >> rf
    ```

### Operations
You can perform much of the same operations as we did for the individual node but now taking into account
everything in the pipeline.

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Nodes"
space = pipeline.search_space("configspace")
config = space.sample_configuration()
configured_pipeline = pipeline.configure(config)
from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide
doc_print(print, config)  # markdown-exec: hide
doc_print(print, configured_pipeline)  # markdown-exec: hide
```

!!! inline end "Other notions of Sequential"

    We'll see this later but wherever we expect a `Node`, for example as an argument to
    `Sequential` or any other type of pipeline component, a list, i.e. `[node_1, node_2]`,
    will automatically be joined together and interpreted as a `Sequential`.

To build a pipeline of nodes, we simply call [`build(builder=)`][amltk.pipeline.Node.build]. We
explicitly pass the builder we want to use, which informs `build()` how to go from the abstract
pipeline definition you've defined to something concrete you can use.
You can find the [available builders here](../reference/pipelines/builders.md).

```python exec="true" source="material-block" html="true" session="Pipeline-Connecting-Nodes"
from sklearn.pipeline import Pipeline as SklearnPipeline

built_pipeline = configured_pipeline.build("sklearn")
assert isinstance(built_pipeline, SklearnPipeline)
print(built_pipeline._repr_html_())  # markdown-exec: hide
```


## Other Building blocks
We saw the basic building block of a `Component` but AutoML-Toolkit also provides support
for some other kinds of building blocks. These building blocks can be attached and joined
together just like a `Component` can and allow for much more complex pipeline structures.

### Choice
A [`Choice`][amltk.pipeline.Choice] is a way to define a choice between multiple
components. This is useful when you want to search over multiple algorithms, which
may each have their own hyperparameters.

We'll start again by creating two nodes:

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
from dataclasses import dataclass

from amltk.pipeline import Component

@dataclass
class ModelA:
    i: int

@dataclass
class ModelB:
    c: str

model_a = Component(ModelA, space={"i": (0, 100)})
model_b = Component(ModelB, space={"c": ["red", "blue"]})
from amltk._doc import doc_print; doc_print(print, model_a, output="html", fontsize="small")  # markdown-exec: hide
doc_print(print, model_b, output="html", fontsize="small")  # markdown-exec: hide
```

Now combining them into a choice is rather straight forward:

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
from amltk.pipeline import Choice

model_choice = Choice(model_a, model_b, name="estimator")
from amltk._doc import doc_print; doc_print(print, model_choice, output="html", fontsize="small")  # markdown-exec: hide
```

Just as we did with a `Component`, we can also get a [`search_space()`][amltk.pipeline.Node.search_space]
from the choice.

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
space = model_choice.search_space("configspace")
from amltk._doc import doc_print; doc_print(print, space, output="html")  # markdown-exec: hide
```

??? warning inline end "Conditionals and Search Spaces"

    Not all search space implementations support conditionals and so some
    `parser=` may not be able to handle this. In this case, there won't be
    any conditionality in the search space.

    Check out the [parser reference](../reference/pipelines/spaces.md)
    for more information.

When we `configure()` a choice, we will collapse it down to a single component. This is
done according to what is set in the config.

```python exec="true" source="material-block" html="true" session="Pipeline-Choice"
config = space.sample_configuration()
configured_choice = model_choice.configure(config)
from amltk._doc import doc_print; doc_print(print, configured_choice, output="html")  # markdown-exec: hide
```
You'll notice that it set the `.config` of the `Choice` to `#!python {"__choice__": "model_a"}` or
`#!python {"__choice__": "model_b"}`. This lets a builder know which of these two to build.

### Split
A [`Split`][amltk.pipeline.Split] is a way to signify a split in the dataflow of a pipeline.
This `Split` by itself will not do anything but it informs the builder about what to do.
Each builder will have if it's own specific strategy for dealing with one.

Let's go ahead with a scikit-learn example, where we'll split the data into categorical
and numerical features and then perform some preprocessing on each of them.

```python exec="true" source="material-block" html="True" session="Pipeline-Split3"
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from amltk.pipeline import Component, Split

select_categories = make_column_selector(dtype_include=object)
select_numerical = make_column_selector(dtype_include=np.number)

preprocessor = Split(
    {
        "categories": [SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(drop="first")],
        "numerics": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
    },
    config={"categories": select_categories, "numerics": select_numerical},
    name="feature_preprocessing",
)
from amltk._doc import doc_print; doc_print(print, preprocessor)  # markdown-exec: hide
```

An important thing to note here is that first, we passed a `dict` to `Split`, such that
we can name the individual paths. This is important because we need some name to refer
to them when configuring the `Split`. It does this by simply wrapping
each of the paths in a [`Sequential`][amltk.pipeline.Sequential].

The second thing is that the parameters set for the `.config` matches those of the
paths. This let's the `Split` know which data should be sent where. Each `builder=`
will have it's own way of how to set up a `Split` and you should refer to
the [builders reference](../reference/pipelines/builders.md) for more information.

Our last step is just to convert this into a useable object and so once again
we use [`build()`][amltk.pipeline.Node.build].

```python exec="true" source="material-block" html="True" session="Pipeline-Split3"
built_pipeline = preprocessor.build("sklearn")
from amltk._doc import doc_print; doc_print(print, built_pipeline)  # markdown-exec: hide
```

### Join

!!! todo "TODO"

    TODO
 

### Searchable

!!! todo "TODO"

    TODO

### Option

!!! todo "TODO"

    Please feel free to provide a contribution!

