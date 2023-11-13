"""[Optuna](https://optuna.org/) parser for parsing out a
[`search_space()`][amltk.pipeline.node.Node.search_space].
from a pipeline.

!!! tip "Requirements"

    This requires `Optuna` which can be installed with:

    ```bash
    pip install amltk[optuna]

    # Or directly
    pip install optuna
    ```

??? warning "Limitations"

    Optuna feature a very dynamic search space (_define-by-run_),
    where people typically sample from some trial object and use traditional
    python control flow to define conditionality.

    This means we can not trivially represent this conditionality in a static
    search space. While _band-aids_ are possible,
    it naturally does not sit well with the static output of a parser.

    As such, our parser **does not support conditionals or choices!**.
    Users may still use the _define-by-run_ within their optimization function
    itself.

    If you have experience with Optuna and have any suggestions,
    please feel free to open an issue or PR on GitHub!

### Usage
The typical way to represent a search space for Optuna is just to use a dictionary,
where the keys are the names of the hyperparameters and the values are either
integer/float tuples indicating boundaries or some discrete set of values.
It is possible to have the value directly be a
`BaseDistribution`, an optuna type, when you need to customize the distribution more.


```python exec="true" source="material-block" html="true" session="optuna-parser"
from amltk.pipeline import Component
from optuna.distributions import FloatDistribution

c = Component(
    object,
    space={
        "myint": (1, 10),
        "myfloat": (1.0, 10.0),
        "mycategorical": ["a", "b", "c"],
        "log-scale-custom": FloatDistribution(1e-10, 1e-2, log=True),
    },
    name="name",
)

space = c.search_space(parser="optuna")
from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide
```

You may also just pass the `parser=` function directly if preferred

```python exec="true" source="material-block" html="true" session="optuna-parser"
from amltk.pipeline.parsers.optuna import parser as optuna_parser

space = c.search_space(parser=optuna_parser)
from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide
```

When using [`search_space()`][amltk.pipeline.node.Node.search_space] on a some nested
structures, you may want to flatten the names of the hyperparameters. For this you
can use `flat=`

```python exec="true" source="material-block" html="true"
from amltk.pipeline import Searchable, Sequential

seq = Sequential(
    Searchable({"myint": (1, 10)}, name="nested_1"),
    Searchable({"myfloat": (1.0, 10.0)}, name="nested_2"),
    name="seq"
)

hierarchical_space = seq.search_space(parser="optuna", flat=False)  # Default
from amltk._doc import doc_print; doc_print(print, hierarchical_space)  # markdown-exec: hide

flat_space = seq.search_space(parser="optuna", flat=False)  # Default
from amltk._doc import doc_print; doc_print(print, flat_space)  # markdown-exec: hide
```

"""  # noqa: E501

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from amltk._functional import prefix_keys

if TYPE_CHECKING:
    from typing import TypeAlias

    from amltk.pipeline import Node

    OptunaSearchSpace: TypeAlias = dict[str, BaseDistribution]

PAIR = 2


def _convert_hp_to_optuna_distribution(
    name: str,
    hp: tuple | Sequence | CategoricalChoiceType | BaseDistribution,
) -> BaseDistribution:
    match hp:
        case BaseDistribution():
            return hp
        case None | bool() | int() | str() | float():
            return CategoricalDistribution([hp])
        case tuple() as tup if len(tup) == PAIR:
            match tup:
                case (int() | np.integer(), int() | np.integer()):
                    x, y = tup
                    return IntDistribution(int(x), int(y))
                case (float() | np.floating(), float() | np.floating()):
                    x, y = tup
                    return FloatDistribution(float(x), float(y))
                case (x, y):
                    raise ValueError(
                        f"Expected {name} to have same type for lower and upper bound,"
                        f"got lower: {type(x)}, upper: {type(y)}.",
                    )
        case Sequence():
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            return CategoricalDistribution(hp)
        case _:
            raise ValueError(
                f"Could not parse {name} as a valid Optuna distribution." f"\n{hp=}",
            )

    raise ValueError(f"Could not parse {name} as a valid Optuna distribution.\n{hp=}")


def _parse_space(node: Node) -> OptunaSearchSpace:
    match node.space:
        case None:
            space = {}
        case Mapping():
            space = {
                name: _convert_hp_to_optuna_distribution(name=name, hp=hp)
                for name, hp in node.space.items()
            }
        case _:
            raise ValueError(
                f"Can only parse mappings with Optuna but got {node.space=}",
            )

    if node.config is not None:
        for name, value in node.config.items():
            if name in space:
                space[name] = CategoricalDistribution([value])

    return space


def parser(
    node: Node,
    *,
    flat: bool = False,
    conditionals: bool = False,
    delim: str = ":",
) -> OptunaSearchSpace:
    """Parse a Node and its children into a ConfigurationSpace.

    Args:
        node: The Node to parse
        flat: Whether to have a hierarchical naming scheme for nodes and their children.
        conditionals: Whether to include conditionals in the space from a
            [`Choice`][amltk.pipeline.Choice]. If this is `False`, this will
            also remove all forbidden clauses and other conditional clauses.
            The primary use of this functionality is that some optimizers do not
            support these features.

            !!! TODO "Not yet supported"

                This functionality is not yet supported as we can't encode this into
                a static Optuna search space.

        delim: The delimiter to use for the names of the hyperparameters.
    """
    if conditionals:
        raise NotImplementedError("Conditionals are not yet supported with Optuna.")

    space = prefix_keys(_parse_space(node), prefix=f"{node.name}{delim}")

    for child in node.nodes:
        subspace = parser(child, flat=flat, conditionals=conditionals, delim=delim)
        if not flat:
            subspace = prefix_keys(subspace, prefix=f"{node.name}{delim}")

        for name, hp in subspace.items():
            if name in space:
                raise ValueError(
                    f"Duplicate name {name} already in space from space of {node.name}",
                    f"\nCurrently parsed space: {space}",
                )
            space[name] = hp

    return space
