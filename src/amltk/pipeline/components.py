"""The provided subclasses of a [`Node`][amltk.pipeline.node.Node]
that can be used can be assembled into a pipeline.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload
from typing_extensions import Self, override

from more_itertools import first_true

from amltk._functional import entity_name, mapping_select
from amltk.exceptions import (
    ComponentBuildError,
    NoChoiceMadeError,
    NodeNotFoundError,
)
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
class Component(Node[Item, Space]):
    """A [`Component`][amltk.pipeline.Component] of the pipeline with
    a possible item and **no children**.

    This is the basic building block of most pipelines, it accepts
    as it's [`item=`][amltk.pipeline.node.Node.item] some function that will be
    called with [`build_item()`][amltk.pipeline.components.Component.build_item] to
    build that one part of the pipeline.

    When [`build_item()`][amltk.pipeline.Component.build_item] is called, whatever
    the config of the component is at that time, will be used to construct the item.

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
        """Initialize a component.

        Args:
            item: The item attached to this node.
            name: The name of the node. If not specified, the name will be
                generated from the item.
            config: The configuration for this node.
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
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
        try:
            return self.item(**{**config, **kwargs})
        except TypeError as e:
            new_msg = f"Failed to build `{self.item=}` with `{self.config=}`.\n"
            if any(kwargs):
                new_msg += f"Extra {kwargs=} were also provided.\n"
            new_msg += (
                "If the item failed to initialize, a common reason can be forgetting"
                " to call `configure()` on the `Component` or the pipeline it is in or"
                " not calling `build()`/`build_item()` on the **returned** value of"
                " `configure()`.\n"
                "Reasons may also include not having fully specified the `config`"
                " initially, it having not being configured fully from `configure()`"
                " or from misspecfying parameters in the `space`."
            )
            raise ComponentBuildError(new_msg) from e


@dataclass(init=False, frozen=True, eq=True)
class Searchable(Node[None, Space]):  # type: ignore
    """A [`Searchable`][amltk.pipeline.Searchable]
    node of the pipeline which just represents a search space, no item attached.

    While not usually applicable to pipelines you want to build, this node
    is useful for creating a search space, especially if the real pipeline you
    want to optimize can not be built directly. For example, if you are optimize
    a script, you may wish to use a `Searchable` to represent the search space
    of that script.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Searchable

    script_space = Searchable({"mode": ["orange", "blue", "red"], "n": (10, 100)})
    from amltk._doc import doc_print; doc_print(print, script_space)  # markdown-exec: hide
    ```

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    item: None = None
    """A searchable has no item."""

    nodes: tuple[()] = ()
    """A searchable has no children."""

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
        """Initialize a choice.

        Args:
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            name: The name of the node. If not specified, a random one will
                be generated.
            config: The configuration for this node. Useful for setting some
                default values.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
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

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    item: Item
    """The fixed item that this node represents."""

    space: None = None
    """A fixed node has no search space."""

    fidelities: None = None
    """A fixed node has no search space."""

    config: None = None
    """A fixed node has no config."""

    config_transform: None = None
    """A fixed node has no config so no transform."""

    nodes: tuple[()] = ()
    """A fixed node has no children."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(panel_color="#56351E")

    _NODES_INIT: ClassVar = None

    def __init__(  # noqa: D417
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
        """Initialize a fixed node.

        Args:
            item: The item attached to this node. Will be fixed and can not
                be configured.
            name: The name of the node. If not specified, the name will be
                generated from the item.
            meta: Any meta information about this node.
        """
        super().__init__(
            name=name if name is not None else entity_name(item),
            item=item,
            config=config,
            space=space,
            fidelities=fidelities,
            config_transform=config_transform,
            meta=meta,
        )


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
    ```

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    nodes: tuple[Node, ...]
    """The nodes ordered in series."""

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
        """Initialize a sequential node.

        Args:
            nodes: The nodes that this node leads to. In the case of a `Sequential`,
                the order here matters and it signifies that data should first
                be passed through the first node, then the second, etc.
            item: The item attached to this node (if any).
            name: The name of the node. If not specified, the name will be
                randomly generated.
            config: The configuration for this node.
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
        _nodes = tuple(as_node(n) for n in nodes)

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
    ```

    !!! warning "Order of nodes"

        The given nodes of a choice are always ordered according
        to their name, so indexing `choice.nodes` may not be reliable
        if modifying the choice dynamically.

        Please use `choice["name"]` to access the nodes instead.

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

    nodes: tuple[Node, ...]
    """The choice of possible nodes that this choice could take."""

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
        """Initialize a choice node.

        Args:
            nodes: The nodes that should be chosen between for this node.
            item: The item attached to this node (if any).
            name: The name of the node. If not specified, the name will be
                randomly generated.
            config: The configuration for this node.
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
        _nodes: tuple[Node, ...] = tuple(
            sorted((as_node(n) for n in nodes), key=lambda n: n.name),
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

    @override
    def configure(
        self,
        config: Config,
        *,
        prefixed_name: bool | None = None,
        transform_context: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Self:
        """Configure this node and anything following it with the given config.

        !!! note "Configuring a choice"

            For a Choice, if the config has a `__choice__` key, then only the node
            chosen will be configured. The others will not be configured at all and
            their config will be discarded.

        Args:
            config: The configuration to apply
            prefixed_name: Whether items in the config are prefixed by the names
                of the nodes.
                * If `None`, the default, then `prefixed_name` will be assumed to
                    be `True` if this node has a next node or if the config has
                    keys that begin with this nodes name.
                * If `True`, then the config will be searched for items prefixed
                    by the name of the node (and subsequent chained nodes).
                * If `False`, then the config will be searched for items without
                    the prefix, i.e. the config keys are exactly those matching
                    this nodes search space.
            transform_context: Any context to give to `config_transform=` of individual
                nodes.
            params: The params to match any requests when configuring this node.
                These will match against any ParamRequests in the config and will
                be used to fill in any missing values.

        Returns:
            The configured node
        """
        # Get the config for this node
        match prefixed_name:
            case True:
                config = mapping_select(config, f"{self.name}:")
            case False:
                pass
            case None if any(k.startswith(f"{self.name}:") for k in config):
                config = mapping_select(config, f"{self.name}:")
            case None:
                pass

        _kwargs: dict[str, Any] = {}

        # Configure all the branches if exists
        # This part is what differs for a Choice
        if len(self.nodes) > 0:
            choice_made = config.get("__choice__", None)
            if choice_made is not None:
                matching_child = first_true(
                    self.nodes,
                    pred=lambda node: node.name == choice_made,
                    default=None,
                )
                if matching_child is None:
                    raise ValueError(
                        f"Can not find matching child for choice {self.name} with child"
                        f" {choice_made}."
                        "\nPlease check the config and ensure that the choice is one of"
                        f" {[n.name for n in self.nodes]}."
                        f"\nThe config recieved at this choice node was {config=}.",
                    )

                # We still iterate over all of them just to ensure correct ordering
                nodes = tuple(
                    node.copy()
                    if node.name != choice_made
                    else matching_child.configure(
                        config,
                        prefixed_name=True,
                        transform_context=transform_context,
                        params=params,
                    )
                    for node in self.nodes
                )
                _kwargs["nodes"] = nodes
            else:
                nodes = tuple(
                    node.configure(
                        config,
                        prefixed_name=True,
                        transform_context=transform_context,
                        params=params,
                    )
                    for node in self.nodes
                )
                _kwargs["nodes"] = nodes

        this_config = {
            hp: v
            for hp, v in config.items()
            if (
                ":" not in hp
                and not any(hp.startswith(f"{node.name}") for node in self.nodes)
            )
        }
        if self.config is not None:
            this_config = {**self.config, **this_config}

        this_config = dict(self._fufill_param_requests(this_config, params=params))

        if self.config_transform is not None:
            this_config = dict(self.config_transform(this_config, transform_context))

        if len(this_config) > 0:
            _kwargs["config"] = dict(this_config)

        return self.mutate(**_kwargs)


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
            "categories": make_column_selector(dtype_include="category"),
            "numerical": make_column_selector(dtype_exclude="category"),
        },
        name="my_split"
    )
    from amltk._doc import doc_print; doc_print(print, preprocessor)  # markdown-exec: hide
    ```

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """  # noqa: E501

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
        """Initialize a split node.

        Args:
            nodes: The nodes that this node leads to. You may also provide
                a dictionary where the keys are the names of the nodes and
                the values are the nodes or list of nodes themselves.
            item: The item attached to this node. The object created by `item`
                should be capable of figuring out how to deal with its child nodes.
            name: The name of the node. If not specified, the name will be
                generated from the item.
            config: The configuration for this split.
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
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
    ```

    See Also:
        * [`Node`][amltk.pipeline.node.Node]
    """

    nodes: tuple[Node, ...]
    """The nodes that should be joined together in parallel."""

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
        """Initialize a join node.

        Args:
            nodes: The nodes that should be joined together in parallel.
            item: The item attached to this node (if any).
            name: The name of the node. If not specified, the name will be
                randomly generated.
            config: The configuration for this node.
            space: The search space for this node. This will be used when
                [`search_space()`][amltk.pipeline.node.Node.search_space] is called.
            fidelities: The fidelities for this node.
            config_transform: A function that transforms the `config=` parameter
                during [`configure(config)`][amltk.pipeline.node.Node.configure]
                before return the new configured node. Useful for times where
                you need to combine multiple parameters into one.
            meta: Any meta information about this node.
        """
        _nodes = tuple(as_node(n) for n in nodes)

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
