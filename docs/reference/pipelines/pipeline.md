A pipeline is a collection of [`Node`][amltk.pipeline.node.Node]s
that are connected together to form a directed acylic graph, where the nodes
follow a parent-child relation ship. The purpose of these is to form some _abstract_
representation of what you want to search over/optimize and then build into a concrete object.

## Key Operations
Once a pipeline is created, you can perform 3 very critical operations on it:

* [`search_space(parser=...)`][amltk.pipeline.node.Node.search_space] - This will return the
  search space of the pipeline, as defined by it's nodes. You can find the reference to
  the [available parsers and search spaces here](../pipelines/spaces.md).
* [`configure(config=...)`][amltk.pipeline.node.Node.configure] - This will return a
  new pipeline where each node is configured correctly.
* [`build(builder=...)`][amltk.pipeline.node.Node.build] - This will return some
    concrete object from a configured pipeline. You can find the reference to
    the [available builders here](../pipelines/builders.md).

## Node
A [`Node`][amltk.pipeline.node.Node] is the basic building block of a pipeline.
It contains various attributes, such as a

- [`.name`][amltk.pipeline.node.Node.name] - The name of the node, which is used
    to identify it in the pipeline.
- [`.item`][amltk.pipeline.node.Node.item] - The concrete object or some function to construct one
- [`.space`][amltk.pipeline.node.Node.space] - A search space to consider for this node
- [`.config`][amltk.pipeline.node.Node.config] - The specific configuration to use for this
    node once `build` is called.
- [`.nodes`][amltk.pipeline.node.Node.nodes] - Other nodes that this node links to.

To give syntactic meaning to these nodes, we have various subclasses. For example,
[`Sequential`][amltk.pipeline.components.Sequential] is a node where the order of the
`nodes` it contains matter, while a [`Component`][amltk.pipeline.components.Component] is a node
that can be used to parametrize and construct a concrete object, but does not lead to anything else.

Each node type here is either a _leaf_ or a _branch_, where a _branch_ has children, while
while a _leaf_ does not.

There various components are listed here:

### [`Component`][amltk.pipeline.Component] - `leaf`
A parametrizable node type with some way to build an object, given a configuration.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Component
from dataclasses import dataclass

@dataclass
class Model:
    x: float

c = Component(Model, space={"x": (0.0, 1.0)}, name="model")
from amltk._doc import doc_print; doc_print(print, c) # markdown-exec: hide
```

### [`Searchable`][amltk.pipeline.Searchable] - `leaf`
A parametrizable node type that contains a search space that should be searched over,
but does not provide a concrete object.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Searchable

def run_script(mode, n):
    # ... run some actual script
    pass

script_space = Searchable({"mode": ["orange", "blue", "red"], "n": (10, 100)})
from amltk._doc import doc_print; doc_print(print, script_space)  # markdown-exec: hide
```

### [`Fixed`][amltk.pipeline.Fixed] - `leaf`
A _non-parametrizable_ node type that contains an object that should be used as is.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Component, Fixed, Sequential
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier()
# ... pretend it was fit
fitted_estimator = Fixed(estimator)
from amltk._doc import doc_print; doc_print(print, fitted_estimator)  # markdown-exec: hide
```

### [`Sequential`][amltk.pipeline.Sequential] - `branch`
A node type which signifies an order between its children, such as a sequential
set of preprocessing and estimator through which the data should flow.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Component, Sequential
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

pipeline = Sequential(
    PCA(n_components=3),
    Component(RandomForestClassifier, space={"n_estimators": (10, 100)}),
    name="my_pipeline"
)
from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
```

### [`Choice`][amltk.pipeline.Choice] - `branch`
A node type that signifies a choice between multiple children, usually chosen during configuration.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Choice, Component
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
mlp = Component(MLPClassifier, space={"activation": ["logistic", "relu", "tanh"]})

estimator_choice = Choice(rf, mlp, name="estimator")
from amltk._doc import doc_print; doc_print(print, estimator_choice)  # markdown-exec: hide
```

### [`Split`][amltk.pipeline.Split] - `branch`
A node where the output of the previous node is split amongst its children,
according to it's configuration.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Component, Split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector

categorical_pipeline = [
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(drop="first"),
]
numerical_pipeline = Component(SimpleImputer, space={"strategy": ["mean", "median"]})

preprocessor = Split(
    {"categories": categorical_pipeline, "numerical": numerical_pipeline},
    name="my_split"
)
from amltk._doc import doc_print; doc_print(print, preprocessor)  # markdown-exec: hide
```

### [`Join`][amltk.pipeline.Join] - `branch`
A node where the output of the previous node is sent all of its children.

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Join, Component
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

pca = Component(PCA, space={"n_components": (1, 3)})
kbest = Component(SelectKBest, space={"k": (1, 3)})

join = Join(pca, kbest, name="my_feature_union")
from amltk._doc import doc_print; doc_print(print, join)  # markdown-exec: hide
```

## Syntax Sugar
You can connect these nodes together using either the constructors explicitly,
as shown in the examples. We also provide some index operators:

* `>>` - Connect nodes together to form a [`Sequential`][amltk.pipeline.components.Sequential]
* `&` - Connect nodes together to form a [`Join`][amltk.pipeline.components.Join]
* `|` - Connect nodes together to form a [`Choice`][amltk.pipeline.components.Choice]

There is also another short-hand that you may find useful to know:

* `{comp1, comp2, comp3}` - This will automatically be converted into a
    [`Choice`][amltk.pipeline.Choice] between the given components.
* `(comp1, comp2, comp3)` - This will automatically be converted into a
    [`Join`][amltk.pipeline.Join] between the given components.
* `[comp1, comp2, comp3]` - This will automatically be converted into a
    [`Sequential`][amltk.pipeline.Sequential] between the given components.

