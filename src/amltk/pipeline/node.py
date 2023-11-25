"""A pipeline consists of [`Node`][amltk.pipeline.node.Node]s, which hold
the various attributes required to build a pipeline, such as the
[`.item`][amltk.pipeline.node.Node.item], its [`.space`][amltk.pipeline.node.Node.space],
its [`.config`][amltk.pipeline.node.Node.config] and so on.

The [`Node`][amltk.pipeline.node.Node]s are connected to each in a parent-child
relation ship where the children are simply the [`.nodes`][amltk.pipeline.node.Node.nodes]
that the parent leads to.

To give these attributes and relations meaning, there are various subclasses
of [`Node`][amltk.pipeline.node.Node] which give different syntactic meanings
when you want to construct something like a
[`search_space()`][amltk.pipeline.node.Node.search_space] or
[`build()`][amltk.pipeline.node.Node.build] some concrete object out of the
pipeline.

For example, a [`Sequential`][amltk.pipeline.Sequential] node
gives the meaning that each of its children in
[`.nodes`][amltk.pipeline.node.Node.nodes] should follow one another while
something like a [`Choice`][amltk.pipeline.Choice]
gives the meaning that only one of its children should be chosen.

You will likely never have to create a [`Node`][amltk.pipeline.node.Node]
directly, but instead use the various components to create the pipeline.

??? note "Hashing"

    When hashing a node, i.e. to put it in a `set` or as a key in a `dict`,
    only the name of the node and the hash of its children is used.
    This means that two nodes with the same connectivity will be equalling hashed,

??? note "Equality"

    When considering equality, this will be done by comparing all the fields
    of the node. This include even the `parent` and `branches` fields. This
    means two nodes are considered equal if they look the same and they are
    connected in to nodes that also look the same.
"""  # noqa: E501
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeAlias,
    TypeVar,
    overload,
)
from typing_extensions import override

from more_itertools import first_true
from rich.text import Text
from sklearn.pipeline import Pipeline as SklearnPipeline

from amltk._functional import classname, mapping_select, prefix_keys
from amltk._richutil import RichRenderable
from amltk.exceptions import RequestNotMetError
from amltk.types import Config, Item, Space

if TYPE_CHECKING:
    from typing_extensions import Self

    from ConfigSpace import ConfigurationSpace
    from rich.console import RenderableType
    from rich.panel import Panel

    from amltk.pipeline.components import Choice, Join, Sequential
    from amltk.pipeline.parsers.optuna import OptunaSearchSpace

    NodeLike: TypeAlias = (
        set["Node" | "NodeLike"]
        | tuple["Node" | "NodeLike", ...]
        | list["Node" | "NodeLike"]
        | Callable[..., Item]
        | Item
    )

    SklearnPipelineT = TypeVar("SklearnPipelineT", bound=SklearnPipeline)

T = TypeVar("T")
ParserOutput = TypeVar("ParserOutput")
BuilderOutput = TypeVar("BuilderOutput")
P = ParamSpec("P")


_NotSet = object()


class RichOptions(NamedTuple):
    """Options for rich printing."""

    panel_color: str = "default"
    node_orientation: Literal["horizontal", "vertical"] = "horizontal"


@dataclass(frozen=True)
class ParamRequest(Generic[T]):
    """A parameter request for a node. This is most useful for things like seeds."""

    key: str
    """The key to request under."""

    default: T | object = _NotSet
    """The default value to use if the key is not found.

    If left as `_NotSet` (default) then an error will be raised if the
    parameter is not found during configuration with
    [`configure()`][amltk.pipeline.node.Node.configure].
    """

    @property
    def has_default(self) -> bool:
        """Whether this request has a default value."""
        return self.default is not _NotSet


def request(key: str, default: T | object = _NotSet) -> ParamRequest[T]:
    """Create a new parameter request.

    Args:
        key: The key to request under.
        default: The default value to use if the key is not found.
            If left as `_NotSet` (default) then the key will be removed from the
            config once [`configure`][amltk.pipeline.Node.configure] is called and
            nothing has been provided.
    """
    return ParamRequest(key=key, default=default)


@dataclass(init=False, frozen=True, eq=True)
class Node(RichRenderable, Generic[Item, Space]):
    """The core node class for the pipeline.

    These are simple objects that are named and linked together to form
    a chain. They are then wrapped in a `Pipeline` object to provide
    a convenient interface for interacting with the chain.
    """

    name: str = field(hash=True)
    """Name of the node"""

    item: Callable[..., Item] | Item | None = field(hash=False)
    """The item attached to this node"""

    nodes: tuple[Node, ...] = field(hash=True)
    """The nodes that this node leads to."""

    config: Config | None = field(hash=False)
    """The configuration for this node"""

    space: Space | None = field(hash=False)
    """The search space for this node"""

    fidelities: Mapping[str, Any] | None = field(hash=False)
    """The fidelities for this node"""

    config_transform: Callable[[Config, Any], Config] | None = field(hash=False)
    """A function that transforms the configuration of this node"""

    meta: Mapping[str, Any] | None = field(hash=False)
    """Any meta information about this node"""

    _NODES_INIT: ClassVar[Literal["args", "kwargs"] | None] = "args"
    """Whether __init__ takes nodes as positional args, kwargs or does not accept it."""

    RICH_OPTIONS: ClassVar[RichOptions] = RichOptions(
        panel_color="default",
        node_orientation="horizontal",
    )
    """Options for rich printing"""

    def __init__(
        self,
        *nodes: Node,
        name: str,
        item: Item | Callable[[Item], Item] | None = None,
        config: Config | None = None,
        space: Space | None = None,
        fidelities: Mapping[str, Any] | None = None,
        config_transform: Callable[[Config, Any], Config] | None = None,
        meta: Mapping[str, Any] | None = None,
    ):
        """Initialize a choice."""
        super().__init__()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "item", item)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "fidelities", fidelities)
        object.__setattr__(self, "config_transform", config_transform)
        object.__setattr__(self, "meta", meta)
        object.__setattr__(self, "nodes", nodes)

    def __getitem__(self, key: str) -> Node:
        """Get the node with the given name."""
        found = first_true(
            self.nodes,
            None,
            lambda node: node.name == key,
        )
        if found is None:
            raise KeyError(
                f"Could not find node with name {key} in '{self.name}'."
                f" Available nodes are: {', '.join(node.name for node in self.nodes)}",
            )

        return found

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        return (
            self.name == other.name
            and self.item == other.item
            and self.config == other.config
            and self.space == other.space
            and self.fidelities == other.fidelities
            and self.config_transform == other.config_transform
            and self.meta == other.meta
            and self.nodes == other.nodes
        )

    def __or__(self, other: Node | NodeLike) -> Choice:
        from amltk.pipeline.components import as_node

        return as_node({self, as_node(other)})

    def __and__(self, other: Node | NodeLike) -> Join:
        from amltk.pipeline.components import as_node

        return as_node((self, other))  # type: ignore

    def __rshift__(self, other: Node | NodeLike) -> Sequential:
        from amltk.pipeline.components import as_node

        return as_node([self, other])  # type: ignore

    def configure(
        self,
        config: Config,
        *,
        prefixed_name: bool | None = None,
        transform_context: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Self:
        """Configure this node and anything following it with the given config.

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
        if len(self.nodes) > 0:
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

    def fidelity_space(self) -> dict[str, Any]:
        """Get the fidelities for this node and any connected nodes."""
        fids = {}
        for node in self.nodes:
            fids.update(prefix_keys(node.fidelity_space(), f"{self.name}:"))

        return fids

    def linearized_fidelity(self, value: float) -> dict[str, int | float | Any]:
        """Get the liniearized fidelities for this node and any connected nodes.

        Args:
            value: The value to linearize. Must be between [0, 1]

        Return:
            dictionary from key to it's linearized fidelity.
        """
        assert 1.0 <= value <= 100.0, f"{value=} not in [1.0, 100.0]"  # noqa: PLR2004
        d = {}
        for node in self.nodes:
            node_fids = prefix_keys(
                node.linearized_fidelity(value),
                f"{self.name}:",
            )
            d.update(node_fids)

        if self.fidelities is None:
            return d

        for f_name, f_range in self.fidelities.items():
            match f_range:
                case (int() | float(), int() | float()):
                    low, high = f_range
                    fid = low + (high - low) * value
                    fid = low + (high - low) * (value - 1) / 100
                    fid = fid if isinstance(low, float) else round(fid)
                    d[f_name] = fid
                case _:
                    raise ValueError(
                        f"Invalid fidelities to linearize {f_range} for {f_name}"
                        f" in {self}. Only supports ranges of the form (low, high)",
                    )

        return prefix_keys(d, f"{self.name}:")

    def iter(self) -> Iterator[Node]:
        """Iterate the the nodes, including this node.

        Yields:
            The nodes connected to this node
        """
        yield self
        for node in self.nodes:
            yield from node.iter()

    def mutate(self, **kwargs: Any) -> Self:
        """Mutate the node with the given keyword arguments.

        Args:
            **kwargs: The keyword arguments to mutate

        Returns:
            Self
                The mutated node
        """
        _args = ()
        _kwargs = {**self.__dict__, **kwargs}

        # If there's nodes in kwargs, we have to check if it's
        # a positional or keyword argument and handle accordingly.
        if (nodes := _kwargs.pop("nodes", None)) is not None:
            match self._NODES_INIT:
                case "args":
                    _args = nodes
                case "kwargs":
                    _kwargs["nodes"] = nodes
                case None if len(nodes) == 0:
                    pass  # Just ignore it, it's popped out
                case None:
                    raise ValueError(
                        "Cannot mutate nodes when __init__ does not accept nodes",
                    )

        # If there's a config in kwargs, we have to check if it's actually got values
        config = _kwargs.pop("config", None)
        if config is not None and len(config) > 0:
            _kwargs["config"] = config

        # Lastly, we remove anything that can't be passed to kwargs of the
        # subclasses __init__
        _available_kwargs = inspect.signature(self.__init__).parameters.keys()  # type: ignore
        for k in list(_kwargs.keys()):
            if k not in _available_kwargs:
                _kwargs.pop(k)

        return self.__class__(*_args, **_kwargs)

    def copy(self) -> Self:
        """Copy this node, removing all links in the process."""
        return self.mutate()

    def path_to(self, key: str | Node | Callable[[Node], bool]) -> list[Node] | None:
        """Find a path to the given node.

        Args:
            key: The key to search for or a function that returns True if the node
                is the desired node

        Returns:
            The path to the node if found, else None
        """
        # We found our target, just return now

        match key:
            case Node():
                pred = lambda node: node == key
            case str():
                pred = lambda node: node.name == key
            case _:
                pred = key

        for path, node in self.walk():
            if pred(node):
                return path

        return None

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

        for node in self.nodes:
            yield from node.walk(path=[*path, self])

    @overload
    def find(self, key: str | Node | Callable[[Node], bool], default: T) -> Node | T:
        ...

    @overload
    def find(self, key: str | Node | Callable[[Node], bool]) -> Node | None:
        ...

    def find(
        self,
        key: str | Node | Callable[[Node], bool],
        default: T | None = None,
    ) -> Node | T | None:
        """Find a node in that's nested deeper from this node.

        Args:
            key: The key to search for or a function that returns True if the node
                is the desired node
            default: The value to return if the node is not found. Defaults to None

        Returns:
            The node if found, otherwise the default value. Defaults to None
        """
        itr = self.iter()
        match key:
            case Node():
                return first_true(itr, default, lambda node: node == key)
            case str():
                return first_true(itr, default, lambda node: node.name == key)
            case _:
                return first_true(itr, default, key)  # type: ignore

    @overload
    def search_space(
        self,
        parser: Literal["configspace"],
        *,
        flat: bool = False,
        conditionals: bool = True,
        delim: str = ":",
    ) -> ConfigurationSpace:
        ...

    @overload
    def search_space(
        self,
        parser: Literal["optuna"],
        *,
        seed: int | None = None,
        flat: bool = False,
        conditionals: bool = True,
        delim: str = ":",
    ) -> OptunaSearchSpace:
        ...

    @overload
    def search_space(
        self,
        parser: Callable[Concatenate[Node, P], ParserOutput],
        *parser_args: P.args,
        **parser_kwargs: P.kwargs,
    ) -> ParserOutput:
        ...

    def search_space(
        self,
        parser: (
            Callable[Concatenate[Node, P], ParserOutput]
            | Literal["configspace", "optuna"]
        ),
        *parser_args: P.args,
        **parser_kwargs: P.kwargs,
    ) -> ParserOutput | ConfigurationSpace | OptunaSearchSpace:
        """Get the search space for this node.

        Args:
            parser: The parser to use. This can be a function that takes in
                the node and returns the search space or a string that is one of:

                * `#!python "configspace"`: Build a
                    [`ConfigSpace.ConfigurationSpace`](https://automl.github.io/ConfigSpace/master/)
                    out of this node.
                * `#!python "optuna"`: Build a dict of hyperparameters that Optuna can
                    use in its [ask and tell methods](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html#define-and-run)

            parser_args: The positional arguments to pass to the parser
            parser_kwargs: The keyword arguments to pass to the parser

        Returns:
            The search space
        """
        match parser:
            case "configspace":
                from amltk.pipeline.parsers.configspace import parser as cs_parser

                return cs_parser(self, *parser_args, **parser_kwargs)  # type: ignore
            case "optuna":
                from amltk.pipeline.parsers.optuna import parser as optuna_parser

                return optuna_parser(self, *parser_args, **parser_kwargs)  # type: ignore
            case str():  # type: ignore
                raise ValueError(
                    f"Invalid str for parser {parser}. "
                    "Please use 'configspace' or 'optuna' or pass in your own"
                    " parser function",
                )
            case _:
                return parser(self, *parser_args, **parser_kwargs)

    @overload
    def build(
        self,
        builder: Literal["sklearn"],
        *builder_args: Any,
        pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
        **builder_kwargs: Any,
    ) -> SklearnPipelineT:
        ...

    @overload
    def build(
        self,
        builder: Literal["sklearn"],
        *builder_args: Any,
        **builder_kwargs: Any,
    ) -> SklearnPipeline:
        ...

    @overload
    def build(
        self,
        builder: Callable[Concatenate[Node, P], BuilderOutput],
        *builder_args: P.args,
        **builder_kwargs: P.kwargs,
    ) -> BuilderOutput:
        ...

    def build(
        self,
        builder: Callable[Concatenate[Node, P], BuilderOutput] | Literal["sklearn"],
        *builder_args: P.args,
        **builder_kwargs: P.kwargs,
    ) -> BuilderOutput | SklearnPipeline:
        """Build a concrete object out of this node.

        Args:
            builder: The builder to use. This can be a function that takes in
                the node and returns the object or a string that is one of:

                * `#!python "sklearn"`: Build a
                    [`sklearn.pipeline.Pipeline`][sklearn.pipeline.Pipeline]
                    out of this node.

            builder_args: The positional arguments to pass to the builder
            builder_kwargs: The keyword arguments to pass to the builder

        Returns:
            The built object
        """
        match builder:
            case "sklearn":
                from amltk.pipeline.builders.sklearn import build as _build

                return _build(self, *builder_args, **builder_kwargs)  # type: ignore
            case _:
                return builder(self, *builder_args, **builder_kwargs)

    def _rich_iter(self) -> Iterator[RenderableType]:
        """Iterate the panels for rich printing."""
        yield self.__rich__()
        for node in self.nodes:
            yield from node._rich_iter()

    def _rich_table_items(self) -> Iterator[tuple[RenderableType, ...]]:
        """Get the items to add to the rich table."""
        from rich.pretty import Pretty

        from amltk._richutil import Function

        if self.item is not None:
            if isinstance(self.item, type) or callable(self.item):
                yield "item", Function(self.item, signature="...")
            else:
                yield "item", Pretty(self.item)

        if self.config is not None:
            yield "config", Pretty(self.config)

        if self.space is not None:
            yield "space", Pretty(self.space)

        if self.fidelities is not None:
            yield "fidelity", Pretty(self.fidelities)

        if self.config_transform is not None:
            yield "transform", Function(self.config_transform, signature="...")

        if self.meta is not None:
            yield "meta", Pretty(self.meta)

    def _rich_panel_contents(self) -> Iterator[RenderableType]:
        from rich.table import Table
        from rich.text import Text

        options = self.RICH_OPTIONS

        if panel_contents := list(self._rich_table_items()):
            table = Table.grid(padding=(0, 1), expand=False)
            for tup in panel_contents:
                table.add_row(*tup, style="default")
            table.add_section()
            yield table

        if len(self.nodes) > 0:
            match options:
                case RichOptions(node_orientation="horizontal"):
                    table = Table.grid(padding=(0, 1), expand=False)
                    nodes = [node.__rich__() for node in self.nodes]
                    table.add_row(*nodes)
                    yield table
                case RichOptions(node_orientation="vertical"):
                    first, *rest = self.nodes
                    yield first.__rich__()
                    for node in rest:
                        yield Text("↓", style="bold", justify="center")
                        yield node.__rich__()
                case _:
                    raise ValueError(f"Invalid orientation {options.node_orientation}")

    def display(self, *, full: bool = False) -> RenderableType:
        """Display this node.

        Args:
            full: Whether to display the full node or just a summary
        """
        if not full:
            return self.__rich__()

        from rich.console import Group as RichGroup

        return RichGroup(*self._rich_iter())

    @override
    def __rich__(self) -> Panel:
        from rich.console import Group as RichGroup
        from rich.panel import Panel

        clr = self.RICH_OPTIONS.panel_color
        title = Text.assemble(
            (classname(self), f"{clr} bold"),
            "(",
            (self.name, f"{clr} italic"),
            ")",
            style="default",
            end="",
        )
        contents = list(self._rich_panel_contents())
        _content = contents[0] if len(contents) == 1 else RichGroup(*contents)
        return Panel(
            _content,
            title=title,
            title_align="left",
            border_style=clr,
            expand=False,
        )

    def _fufill_param_requests(
        self,
        config: Config,
        params: Mapping[str, Any] | None = None,
    ) -> Config:
        _params = params or {}
        new_config = dict(config)

        for k, request in config.items():
            match request:
                case ParamRequest(key=request_key) if request_key in _params:
                    new_config[k] = _params[request_key]
                case ParamRequest(default=default) if request.has_default:
                    new_config[k] = default
                case ParamRequest():
                    raise RequestNotMetError(
                        f"Missing {request=} for {self}.\nparams={params}",
                    )
                case _:
                    continue

        return new_config
