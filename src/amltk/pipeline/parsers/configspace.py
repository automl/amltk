"""[ConfigSpace](https://automl.github.io/ConfigSpace/master/) is a library for
representing and sampling configurations for hyperparameter optimization.
It features a straightforward API for defining hyperparameters, their ranges
and even conditional dependencies.

It is generally flexible enough for more complex use cases, even
handling the complex pipelines of [AutoSklearn](https://automl.github.io/auto-sklearn/master/)
and [AutoPyTorch](https://automl.github.io/Auto-PyTorch/master/), large
scale hyperparameter spaces over which to optimize entire
pipelines at a time.

!!! tip "Requirements"

    This requires `ConfigSpace` which can be installed with:

    ```bash
    pip install "amltk[configspace]"

    # Or directly
    pip install ConfigSpace
    ```

In general, you should have the
[ConfigSpace documentation](https://automl.github.io/ConfigSpace/master/)
ready to consult for a full understanding of how to construct
hyperparameter spaces with AMLTK.

#### Basic Usage

You can directly us the [`parser()`][amltk.pipeline.parsers.configspace.parser]
function and pass that into the [`search_space()`][amltk.pipeline.Node.search_space]
method of a [`Node`][amltk.pipeline.Node], however you can also simply provide
`#!python search_space(parser="configspace", ...)` for simplicity.

```python exec="true" result="python" source="material-block" hl_lines="27" session="configspace-parser"
from amltk.pipeline import Component, Choice, Sequential
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

my_pipeline = (
    Sequential(name="Pipeline")
    >> Component(PCA, space={"n_components": (1, 3)})
    >> Choice(
        Component(
            SVC,
            space={"C": (0.1, 10.0)}
        ),
        Component(
            RandomForestClassifier,
            space={"n_estimators": (10, 100), "criterion": ["gini", "log_loss"]},
        ),
        Component(
            MLPClassifier,
            space={
                "activation": ["identity", "logistic", "relu"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
        ),
        name="estimator"
    )
)

space = my_pipeline.search_space("configspace")
print(space)
```

Here we have an example of a few different kinds of hyperparmeters,

* `PCA:n_components` is a integer with a range of 1 to 3, uniform distribution, as specified
    by it's integer bounds in a tuple.
* `SVC:C` is a float with a range of 0.1 to 10.0, uniform distribution, as specified
    by it's float bounds in a tuple.
* `RandomForestClassifier:criterion` is a categorical hyperparameter, with two choices,
    `"gini"` and `"log_loss"`.

There is also a [`Choice`][amltk.pipeline.Choice] node, which is a special node that indicates that
we could choose from one of these estimators. This leads to the conditionals that you
can see in the printed out space.

You may wish to remove all conditionals if an `Optimizer` does not support them, or
you may wish to remove them for other reasons. You can do this by passing
`conditionals=False` to the [`parser()`][amltk.pipeline.parsers.configspace.parser] function.

```python exec="true" result="python" source="material-block" hl_lines="27" session="configspace-parser"
print(my_pipeline.search_space("configspace", conditionals=False))
```

Likewise, you can also remove all heirarchy from the space which may make downstream tasks easier,
by passing `flat=True` to the [`parser()`][amltk.pipeline.parsers.configspace.parser] function.

```python exec="true" result="python" source="material-block" hl_lines="27" session="configspace-parser"
print(my_pipeline.search_space("configspace", flat=True))
```

#### More Specific Hyperparameters
You'll often want to be a bit more specific with your hyperparameters, here we just
show a few examples of how you'd couple your pipelines a bit more towards `ConfigSpace`.

```python exec="true" result="python" source="material-block"
from ConfigSpace import Float, Categorical, Normal
from amltk.pipeline import Searchable

s = Searchable(
    space={
        "lr": Float("lr", bounds=(1e-5, 1.), log=True, default=0.3),
        "balance": Float("balance", bounds=(-1.0, 1.0), distribution=Normal(0.0, 0.5)),
        "color": Categorical("color", ["red", "green", "blue"], weights=[2, 1, 1], default="blue"),
    },
    name="Something-To-Search",
)
print(s.search_space("configspace"))
```

#### Conditional ands Advanced Usage
We will refer you to the
[ConfigSpace documentation](https://automl.github.io/ConfigSpace/master/) for the construction
of these. However once you've constructed a `ConfigurationSpace` and added any forbiddens and
conditionals, you may simply set that as the `.space` attribute.

```python exec="true" result="python" source="material-block" hl_lines="27"
from amltk.pipeline import Component, Choice, Sequential
from ConfigSpace import ConfigurationSpace, EqualsCondition, InCondition

myspace = ConfigurationSpace({"A": ["red", "green", "blue"], "B": (1, 10), "C": (-100.0, 0.0)})
myspace.add_conditions([
    EqualsCondition(myspace["B"], myspace["A"], "red"),  # B is active when A is red
    InCondition(myspace["C"], myspace["A"], ["green", "blue"]), # C is active when A is green or blue
])

component = Component(object, space=myspace, name="MyThing")

parsed_space = component.search_space("configspace")
print(parsed_space)
```

"""  # noqa: E501
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Constant

from amltk.pipeline import Choice, Node


def _remove_hyperparameter(
    name: str,
    space: ConfigurationSpace,
    seed: int | None = None,
) -> ConfigurationSpace:
    if name not in space._hyperparameters:
        raise ValueError(f"{name} not in {space}")

    # Copying conditionals only work on objects and not named entities
    # Seeing as we copy objects and don't use the originals, transfering these
    # to the new objects is a bit tedious, possible but not required at this time
    # ... same goes for forbiddens
    if name in space._conditionals:
        raise ValueError("Can't remove conditionals")
    if any(name == f.hyperparameter.name for f in space.get_forbiddens()):
        raise ValueError("Can't remove forbiddens")

    hps = [deepcopy(hp) for hp in space.get_hyperparameters() if hp.name != name]

    new_space = ConfigurationSpace(seed=seed, name=space.name, meta=space.meta)
    new_space.add_hyperparameters(hps)
    return new_space


def _remove_conditionals(
    space: ConfigurationSpace,
    seed: int | None = None,
) -> ConfigurationSpace:
    new_space = ConfigurationSpace(seed=seed, name=space.name, meta=space.meta)
    new_space.add_hyperparameters(space.values())
    return new_space


def _replace_constants(
    config: Mapping[str, Any],
    space: ConfigurationSpace,
    seed: int | None = None,
) -> ConfigurationSpace:
    for key, value in config.items():
        if key in space._hyperparameters:
            space = _remove_hyperparameter(key, space, seed)

            # These are just restrictions on hyperparameters from ConfigSpace
            match value:
                case bool():
                    space.add_hyperparameter(Constant(key, str(value)))
                case int() | float() | str():
                    space.add_hyperparameter(Constant(key, value))
                case _:
                    raise ValueError(f"Can't handle {value} from {config} as Constant")

    return space


def _parse_space(
    node: Node,
    *,
    conditionals: bool = True,
    seed: int | None = None,
) -> ConfigurationSpace:
    space = node.space
    match space:
        case ConfigurationSpace():
            _space = deepcopy(space)
        case Mapping():
            _space = ConfigurationSpace(dict(space))
        case None:
            _space = ConfigurationSpace()
        case _:
            raise ValueError(f"Can't handle {space} from {node}")

    if not conditionals:
        _space = _remove_conditionals(_space, seed)

    if node.config is not None:
        _space = _replace_constants(node.config, _space, seed)

    if seed is not None:
        _space.seed(seed)

    return _space


def parser(
    node: Node,
    *,
    seed: int | None = None,
    flat: bool = False,
    conditionals: bool = True,
    delim: str = ":",
) -> ConfigurationSpace:
    """Parse a Node and its children into a ConfigurationSpace.

    Args:
        node: The Node to parse
        seed: The seed to use for the ConfigurationSpace
        flat: Whether to have a heirarchical naming scheme for nodes and their children.
        conditionals: Whether to include conditionals in the space from a
            [`Choice`][amltk.pipeline.Choice]. If this is `False`, this will
            also remove all forbidden clauses and other conditional clauses.
            The primary use of this functionality is that some optimizers do not
            support these features.
        delim: The delimiter to use for the names of the hyperparameters
    """
    space = ConfigurationSpace(seed=seed)
    space.add_configuration_space(
        prefix=node.name,
        delimiter=delim,
        configuration_space=_parse_space(node, seed=seed, conditionals=conditionals),
    )

    children = node.nodes

    choice = None
    if isinstance(node, Choice) and any(children):
        choice = Categorical(
            name=f"{node.name}{delim}__choice__",
            items=[child.name for child in children],
        )
        space.add_hyperparameter(choice)

    for child in children:
        space.add_configuration_space(
            prefix=node.name if not flat else "",
            delimiter=delim if not flat else "",
            configuration_space=parser(
                child,
                seed=seed,
                flat=flat,
                conditionals=conditionals,
                delim=delim,
            ),
            parent_hyperparameter=(
                {"parent": choice, "value": child.name}
                if choice and conditionals
                else None
            ),
        )

    return space
