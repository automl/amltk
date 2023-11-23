"""You can use the various different node types to build a pipeline.

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

For each of these components we will show examples using
the [`#! "sklearn"` builder][amltk.pipeline.builders.sklearn.build]

The components are:

### Component

::: amltk.pipeline.components.Component
    options:
        members: false

### Sequential

::: amltk.pipeline.components.Sequential
    options:
        members: false

### Choice

::: amltk.pipeline.components.Choice
    options:
        members: false

### Split

::: amltk.pipeline.components.Split
    options:
        members: false

### Join

::: amltk.pipeline.components.Join
    options:
        members: false

### Fixed

::: amltk.pipeline.components.Fixed
    options:
        members: false

### Searchable

::: amltk.pipeline.components.Searchable
    options:
        members: false
"""  # noqa: E501
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload
from typing_extensions import override

from more_itertools import all_unique, first_true

from amltk._functional import entity_name
from amltk.exceptions import DuplicateNamesError, NoChoiceMadeError, NodeNotFoundError
from amltk.pipeline.node import Node, RichOptions
from amltk.randomness import randuid
from amltk.types import Config, Item, Space

if TYPE_CHECKING:
    from amltk.pipeline.node import NodeLike


T = TypeVar("T")
NodeT = TypeVar("NodeT", bound=Node)


@overload
def as_node(thing: Node, name: str | None = ...) -> Node:  # type: ignore
    ...


@overload
def as_node(thing: tuple[Node | NodeLike, ...], name: str | None = ...) -> Join:  # type: ignore
    ...


@overload
def as_node(thing: set[Node | NodeLike], name: str | None = ...) -> Choice:  # type: ignore
    ...


@overload
def as_node(thing: list[Node | NodeLike], name: str | None = ...) -> Sequential:  # type: ignore
    ...


@overload
def as_node(  # type: ignore
    thing: Callable[..., Item],
    name: str | None = ...,
) -> Component[Item, None]:
    ...


@overload
def as_node(thing: Item, name: str | None = ...) -> Fixed[Item]:
    ...


def as_node(  # noqa: PLR0911
    thing: Node | NodeLike[Item],
    name: str | None = None,
) -> Node | Choice | Join | Sequential | Fixed[Item]:
    """Convert a node, pipeline, set or tuple into a component, copying anything
    in the process and removing all linking to other nodes.

    Args:
        thing: The thing to convert
        name: The name of the node. If it already a node, it will be renamed to that
            one.

    Returns:
        The component
    """
    match thing:
        case set():
            return Choice(*thing, name=name)
        case tuple():
            return Join(*thing, name=name)
        case list():
            return Sequential(*thing, name=name)
        case Node():
            name = thing.name if name is None else name
            return thing.mutate(name=name)
        case type():
            return Component(thing, name=name)
        case thing if (inspect.isfunction(thing) or inspect.ismethod(thing)):
            return Component(thing, name=name)
        case _:
            return Fixed(thing, name=name)


@dataclass(init=False, frozen=True, eq=True)
class Join(Node[Item, Space]):
    """[`Join`][amltk.pipeline.Join] together different parts of the pipeline.

    This indicates the different children in
    [`.nodes`][amltk.pipeline.Node.nodes] should act in tandem with one
    another, for example, concatenating the outputs of the various members of the
    `Join`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Join, Component
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest

    pca = Component(PCA, space={"n_components": (1, 3)})
    kbest = Component(SelectKBest, space={"k": (1, 3)})

    join = Join(pca, kbest, name="my_feature_union")
    from amltk._doc import doc_print; doc_print(print, join)  # markdown-exec: hide

    space = join.search_space("configspace")
    from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide

    pipeline = join.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    You may also just join together nodes using an infix operator `&` if you prefer:

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Join, Component
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest

    pca = Component(PCA, space={"n_components": (1, 3)})
    kbest = Component(SelectKBest, space={"k": (1, 3)})

    # Can not parametrize or name the join
    join = pca & kbest
    from amltk._doc import doc_print; doc_print(print, join)  # markdown-exec: hide

    # With a parametrized join
    join = (
        Join(name="my_feature_union") & pca & kbest
    )
    item = join.build("sklearn")
    print(item._repr_html_())  # markdown-exec: hide
    ```

    Whenever some other node sees a tuple, i.e. `(comp1, comp2, comp3)`, this
    will automatically be converted into a `Join`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Sequential, Component
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import RandomForestClassifier

    pca = Component(PCA, space={"n_components": (1, 3)})
    kbest = Component(SelectKBest, space={"k": (1, 3)})

    # Can not parametrize or name the join
    join = Sequential(
        (pca, kbest),
        RandomForestClassifier(n_estimators=5),
        name="my_feature_union",
    )
    print(join._repr_html_())  # markdown-exec: hide
    ```

    Like all [`Node`][amltk.pipeline.node.Node]s, a `Join` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    nodes: tuple[Node, ...]
    """The nodes that this node leads to."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="#7E6B8F")

    _NODES_INIT: ClassVar = "args"

    def __init__(
        self,
        *nodes: Node | NodeLike,
        name: str | None = None,
        item: Item | Callable[[Item], Item] | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        _nodes = tuple(as_node(n) for n in nodes)
        if not all_unique(_nodes, key=lambda node: node.name):
            raise ValueError(
                f"Can't handle nodes they do not all contain unique names, {nodes=}."
                "\nAll nodes must have a unique name. Please provide a `name=` to them",
            )

        if name is None:
            name = f"Join-{randuid(8)}"

        super().__init__(
            *_nodes,
            name=name,
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )

    @override
    def __and__(self, other: Node | NodeLike) -> Join:
        other_node = as_node(other)
        if any(other_node.name == node.name for node in self.nodes):
            raise ValueError(
                f"Can't handle node with name '{other_node.name} as"
                f" there is already a node called '{other_node.name}' in {self.name}",
            )

        nodes = (*tuple(as_node(n) for n in self.nodes), other_node)
        return self.mutate(name=self.name, nodes=nodes)


@dataclass(init=False, frozen=True, eq=True)
class Choice(Node[Item, Space]):
    """A [`Choice`][amltk.pipeline.Choice] between different subcomponents.

    This indicates that a choice should be made between the different children in
    [`.nodes`][amltk.pipeline.Node.nodes], usually done when you
    [`configure()`][amltk.pipeline.node.Node.configure] with some `config` from
    a [`search_space()`][amltk.pipeline.node.Node.search_space].

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Choice, Component
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
    mlp = Component(MLPClassifier, space={"activation": ["logistic", "relu", "tanh"]})

    estimator_choice = Choice(rf, mlp, name="estimator")
    from amltk._doc import doc_print; doc_print(print, estimator_choice)  # markdown-exec: hide

    space = estimator_choice.search_space("configspace")
    from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide

    config = space.sample_configuration()
    from amltk._doc import doc_print; doc_print(print, config)  # markdown-exec: hide

    configured_choice = estimator_choice.configure(config)
    from amltk._doc import doc_print; doc_print(print, configured_choice)  # markdown-exec: hide

    chosen_estimator = configured_choice.chosen()
    from amltk._doc import doc_print; doc_print(print, chosen_estimator)  # markdown-exec: hide

    estimator = chosen_estimator.build_item()
    from amltk._doc import doc_print; doc_print(print, estimator)  # markdown-exec: hide
    ```

    You may also just add nodes to a `Choice` using an infix operator `|` if you prefer:

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Choice, Component
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
    mlp = Component(MLPClassifier, space={"activation": ["logistic", "relu", "tanh"]})

    estimator_choice = (
        Choice(name="estimator") | mlp | rf
    )
    from amltk._doc import doc_print; doc_print(print, estimator_choice)  # markdown-exec: hide
    ```

    Whenever some other node sees a set, i.e. `{comp1, comp2, comp3}`, this
    will automatically be converted into a `Choice`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Choice, Component, Sequential
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.impute import SimpleImputer

    rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
    mlp = Component(MLPClassifier, space={"activation": ["logistic", "relu", "tanh"]})

    pipeline = Sequential(
        SimpleImputer(fill_value=0),
        {mlp, rf},
        name="my_pipeline",
    )
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    Like all [`Node`][amltk.pipeline.node.Node]s, a `Choice` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    !!! warning "Order of nodes"

        The given nodes of a choice are always ordered according
        to their name, so indexing `choice.nodes` may not be reliable
        if modifying the choice dynamically.

        Please use `choice["name"]` to access the nodes instead.

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    nodes: tuple[Node, ...]
    """The nodes that this node leads to."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="#FF4500")
    _NODES_INIT: ClassVar = "args"

    def __init__(
        self,
        *nodes: Node | NodeLike,
        name: str | None = None,
        item: Item | Callable[[Item], Item] | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        _nodes: tuple[Node, ...] = tuple(
            sorted((as_node(n) for n in nodes), key=lambda n: n.name),
        )
        if not all_unique(_nodes, key=lambda node: node.name):
            raise ValueError(
                f"Can't handle nodes as we can not generate a __choice__ for {nodes=}."
                "\nAll nodes must have a unique name. Please provide a `name=` to them",
            )

        if name is None:
            name = f"Choice-{randuid(8)}"

        super().__init__(
            *_nodes,
            name=name,
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )

    @override
    def __or__(self, other: Node | NodeLike) -> Choice:
        other_node = as_node(other)
        if any(other_node.name == node.name for node in self.nodes):
            raise ValueError(
                f"Can't handle node with name '{other_node.name} as"
                f" there is already a node called '{other_node.name}' in {self.name}",
            )

        nodes = tuple(
            sorted(
                [as_node(n) for n in self.nodes] + [other_node],
                key=lambda n: n.name,
            ),
        )
        return self.mutate(name=self.name, nodes=nodes)

    def chosen(self) -> Node:
        """The chosen branch.

        Returns:
            The chosen branch
        """
        match self.config:
            case {"__choice__": choice}:
                chosen = first_true(
                    self.nodes,
                    pred=lambda node: node.name == choice,
                    default=None,
                )
                if chosen is None:
                    raise NodeNotFoundError(choice, self.name)

                return chosen
            case _:
                raise NoChoiceMadeError(self.name)


@dataclass(init=False, frozen=True, eq=True)
class Sequential(Node[Item, Space]):
    """A [`Sequential`][amltk.pipeline.Sequential] set of operations in a pipeline.

    This indicates the different children in
    [`.nodes`][amltk.pipeline.Node.nodes] should act one after
    another, feeding the output of one into the next.

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

    space = pipeline.search_space("configspace")
    from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide

    configuration = space.sample_configuration()
    from amltk._doc import doc_print; doc_print(print, configuration)  # markdown-exec: hide

    configured_pipeline = pipeline.configure(configuration)
    from amltk._doc import doc_print; doc_print(print, configured_pipeline)  # markdown-exec: hide

    sklearn_pipeline = pipeline.build("sklearn")
    print(sklearn_pipeline._repr_html_())  # markdown-exec: hide
    ```

    You may also just chain together nodes using an infix operator `>>` if you prefer:

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Join, Component, Sequential
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    pipeline = (
        Sequential(name="my_pipeline")
        >> PCA(n_components=3)
        >> Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
    )
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    Whenever some other node sees a list, i.e. `[comp1, comp2, comp3]`, this
    will automatically be converted into a `Sequential`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Choice
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    pipeline_choice = Choice(
        [SimpleImputer(), RandomForestClassifier()],
        [StandardScaler(), MLPClassifier()],
        name="pipeline_choice"
    )
    from amltk._doc import doc_print; doc_print(print, pipeline_choice)  # markdown-exec: hide
    ```

    Like all [`Node`][amltk.pipeline.node.Node]s, a `Sequential` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    nodes: tuple[Node, ...]
    """The nodes in series."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(
        panel_color="#7E6B8F",
        node_orientation="vertical",
    )
    _NODES_INIT: ClassVar = "args"

    def __init__(
        self,
        *nodes: Node | NodeLike,
        name: str | None = None,
        item: Item | Callable[[Item], Item] | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        _nodes = tuple(as_node(n) for n in nodes)

        # Perhaps we need to do a deeper check on this...
        if not all_unique(_nodes, key=lambda node: node.name):
            raise DuplicateNamesError(self)

        if name is None:
            name = f"Seq-{randuid(8)}"

        super().__init__(
            *_nodes,
            name=name,
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )

    @property
    def tail(self) -> Node:
        """The last step in the pipeline."""
        return self.nodes[-1]

    def __len__(self) -> int:
        """Get the number of nodes in the pipeline."""
        return len(self.nodes)

    @override
    def __rshift__(self, other: Node | NodeLike) -> Sequential:
        other_node = as_node(other)
        if any(other_node.name == node.name for node in self.nodes):
            raise ValueError(
                f"Can't handle node with name '{other_node.name} as"
                f" there is already a node called '{other_node.name}' in {self.name}",
            )

        nodes = (*tuple(as_node(n) for n in self.nodes), other_node)
        return self.mutate(name=self.name, nodes=nodes)

    @override
    def walk(
        self,
        path: Sequence[Node] | None = None,
    ) -> Iterator[tuple[list[Node], Node]]:
        """Walk the nodes in this chain.

        Args:
            path: The current path to this node

        Yields:
            The parents of the node and the node itself
        """
        path = list(path) if path is not None else []
        yield path, self

        path = [*path, self]
        for node in self.nodes:
            yield from node.walk(path=path)

            # Append the previous node so that the next node in the sequence is
            # lead to from the previous node
            path = [*path, node]


@dataclass(init=False, frozen=True, eq=True)
class Split(Node[Item, Space]):
    """A [`Split`][amltk.pipeline.Split] of data in a pipeline.

    This indicates the different children in
    [`.nodes`][amltk.pipeline.Node.nodes] should
    act in parallel but on different subsets of data.

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
        {
            "categories": categorical_pipeline,
            "numerical": numerical_pipeline,
        },
        config={
            # This is how you would configure the split for the sklearn builder in particular
            "categories": make_column_selector(dtype_include="category"),
            "numerical": make_column_selector(dtype_exclude="category"),
        },
        name="my_split"
    )
    from amltk._doc import doc_print; doc_print(print, preprocessor)  # markdown-exec: hide

    space = preprocessor.search_space("configspace")
    from amltk._doc import doc_print; doc_print(print, space)  # markdown-exec: hide

    configuration = space.sample_configuration()
    from amltk._doc import doc_print; doc_print(print, configuration)  # markdown-exec: hide

    configured_preprocessor = preprocessor.configure(configuration)
    from amltk._doc import doc_print; doc_print(print, configured_preprocessor)  # markdown-exec: hide

    built_preprocessor = configured_preprocessor.build("sklearn")
    print(built_preprocessor._repr_html_())  # markdown-exec: hide
    ```

    The split is a slight oddity when compared to the other kinds of components in that
    it allows a `dict` as it's first argument, where the keys are the names of the
    different paths through which data will go and the values are the actual nodes that
    will receive the data.

    If nodes are passed in as they are for all other components, usually the name of the
    first node will be important for any builder trying to make sense of how
    to use the `Split`


    Like all [`Node`][amltk.pipeline.node.Node]s, a `Split` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    nodes: tuple[Node, ...]
    """The nodes that this node leads to."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(
        panel_color="#777DA7",
        node_orientation="horizontal",
    )

    _NODES_INIT: ClassVar = "args"

    def __init__(
        self,
        *nodes: Node | NodeLike | dict[str, Node | NodeLike],
        name: str | None = None,
        item: Item | Callable[[Item], Item] | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        if any(isinstance(n, dict) for n in nodes):
            if len(nodes) > 1:
                raise ValueError(
                    "Can't handle multiple nodes with a dictionary as a node.\n"
                    f"{nodes=}",
                )
            _node = nodes[0]
            assert isinstance(_node, dict)

            def _construct(key: str, value: Node | NodeLike) -> Node:
                match value:
                    case list():
                        return Sequential(*value, name=key)
                    case set() | tuple():
                        return as_node(value, name=key)
                    case _:
                        return Sequential(value, name=key)

            _nodes = tuple(_construct(key, value) for key, value in _node.items())
        else:
            _nodes = tuple(as_node(n) for n in nodes)

        if not all_unique(_nodes, key=lambda node: node.name):
            raise ValueError(
                f"Can't handle nodes they do not all contain unique names, {nodes=}."
                "\nAll nodes must have a unique name. Please provide a `name=` to them",
            )

        if name is None:
            name = f"Split-{randuid(8)}"

        super().__init__(
            *_nodes,
            name=name,
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )


@dataclass(init=False, frozen=True, eq=True)
class Component(Node[Item, Space]):
    """A [`Component`][amltk.pipeline.Component] of the pipeline with
    a possible item and **no children**.

    This is the basic building block of most pipelines, it accepts
    as it's [`item=`][amltk.pipeline.node.Node.item] some function that will be
    called with [`build_item()`][amltk.pipeline.components.Component.build_item] to
    build that one part of the pipeline.

    When [`build_item()`][amltk.pipeline.Component.build_item] is called,
    The [`.config`][amltk.pipeline.node.Node.config] on this node will be passed
    to the function to build the item.

    A common pattern is to use a [`Component`][amltk.pipeline.Component] to
    wrap a constructor, specifying the [`space=`][amltk.pipeline.node.Node.space]
    and [`config=`][amltk.pipeline.node.Node.config] to be used when building the
    item.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Component
    from sklearn.ensemble import RandomForestClassifier

    rf = Component(
        RandomForestClassifier,
        config={"max_depth": 3},
        space={"n_estimators": (10, 100)}
    )
    from amltk._doc import doc_print; doc_print(print, rf)  # markdown-exec: hide

    config = {"n_estimators": 50}  # Sample from some space or something
    configured_rf = rf.configure(config)

    estimator = configured_rf.build_item()
    from amltk._doc import doc_print; doc_print(print, estimator)  # markdown-exec: hide
    ```

    Whenever some other node sees a function/constructor, i.e. `RandomForestClassifier`,
    this will automatically be converted into a `Component`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Sequential
    from sklearn.ensemble import RandomForestClassifier

    pipeline = Sequential(RandomForestClassifier, name="my_pipeline")
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    The default `.name` of a component is the name of the class/function that it will
    use. You can explicitly set the `name=` if you want to when constructing the
    component.

    Like all [`Node`][amltk.pipeline.node.Node]s, a `Component` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    item: Callable[..., Item]
    """A node which constructs an item in the pipeline."""

    nodes: tuple[()]
    """A component has no children."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="#E6AF2E")

    _NODES_INIT: ClassVar = None

    def __init__(
        self,
        item: Callable[..., Item],
        *,
        name: str | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        super().__init__(
            name=name if name is not None else entity_name(item),
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )

    def build_item(self, **kwargs: Any) -> Item:
        """Build the item attached to this component.

        Args:
            **kwargs: Any additional arguments to pass to the item

        Returns:
            Item
                The built item
        """
        config = self.config or {}
        return self.item(**{**config, **kwargs})


@dataclass(init=False, frozen=True, eq=True)
class Searchable(Node[None, Space]):  # type: ignore
    """A [`Searchable`][amltk.pipeline.Searchable]
    node of the pipeline which just represents a search space, no item attached.

    While not usually applicable to pipelines you want to build, this component
    is useful for creating a search space, especially if the the real pipeline you
    want to optimize can not be built directly. For example, if you are optimize
    a script, you may wish to use a `Searchable` to represent the search space
    of that script.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Searchable

    script_space = Searchable({"mode": ["orange", "blue", "red"], "n": (10, 100)})
    from amltk._doc import doc_print; doc_print(print, script_space)  # markdown-exec: hide
    ```

    A `Searchable` explicitly does not allow for `item=` to be set, nor can it have
    any children. A `Searchable` accepts an explicit
    [`name=`][amltk.pipeline.node.Node.name],
    [`config=`][amltk.pipeline.node.Node.config],
    [`space=`][amltk.pipeline.node.Node.space],
    [`fidelities=`][amltk.pipeline.node.Node.fidelities],
    [`config_transform=`][amltk.pipeline.node.Node.config_transform] and
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    item: None = None
    """A searchable has no item."""

    nodes: tuple[()] = ()
    """A component has no children."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="light_steel_blue")

    _NODES_INIT: ClassVar = None

    def __init__(
        self,
        space: Space | None = None,
        *,
        name: str | None = None,
        config: Config | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        if name is None:
            name = f"Searchable-{randuid(8)}"

        super().__init__(
            name=name,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )


@dataclass(init=False, frozen=True, eq=True)
class Fixed(Node[Item, None]):  # type: ignore
    """A [`Fixed`][amltk.pipeline.Fixed] part of the pipeline that
    represents something that can not be configured and used directly as is.

    It consists of an [`.item`][amltk.pipeline.node.Node.item] that is fixed,
    non-configurable and non-searchable. It also has no children.

    This is useful for representing parts of the pipeline that are fixed, for example
    if you have a pipeline that is a `Sequential` of nodes, but you want to
    fix the first component to be a `PCA` with `n_components=3`, you can use a `Fixed`
    to represent that.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Component, Fixed, Sequential
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    rf = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})
    pca = Fixed(PCA(n_components=3))

    pipeline = Sequential(pca, rf, name="my_pipeline")
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    Whenever some other node sees an instance of something, i.e. something that can't be
    called, this will automatically be converted into a `Fixed`.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Sequential
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    pipeline = Sequential(
        PCA(n_components=3),
        RandomForestClassifier(n_estimators=50),
        name="my_pipeline",
    )
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

    The default `.name` of a component is the class name of the item that it will
    use. You can explicitly set the `name=` if you want to when constructing the
    component.

    A `Fixed` accepts only an explicit [`name=`][amltk.pipeline.node.Node.name],
    [`item=`][amltk.pipeline.node.Node.item],
    [`meta=`][amltk.pipeline.node.Node.meta].

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    item: Item = field()
    """The fixed item that this node represents."""

    space: None = None
    """A frozen node has no search space."""

    fidelities: None = None
    """A frozen node has no search space."""

    config: None = None
    """A frozen node has no config."""

    config_transform: None = None
    """A frozen node has no config so no transform."""

    nodes: tuple[()] = ()
    """A component has no children."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="#56351E")

    _NODES_INIT: ClassVar = None

    def __init__(
        self,
        item: Item,
        *,
        name: str | None = None,
        config: None = None,
        space: None = None,
        fidelities: None = None,
        config_transform: None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """See [`Node`][amltk.pipeline.node.Node] for details."""
        super().__init__(
            name=name if name is not None else entity_name(item),
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )
