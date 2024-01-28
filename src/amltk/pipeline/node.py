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
# ruff: noqa: PLR0913
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)
from typing_extensions import override

from more_itertools import first_true
from sklearn.pipeline import Pipeline as SklearnPipeline

from amltk._functional import classname, funcname, mapping_select, prefix_keys
from amltk._richutil import RichRenderable
from amltk.exceptions import RequestNotMetError
from amltk.optimization.history import History
from amltk.optimization.optimizer import Optimizer
from amltk.scheduling import Task
from amltk.scheduling.plugins import Plugin
from amltk.store import PathBucket
from amltk.types import Config, Item, Seed, Space

if TYPE_CHECKING:
    from typing_extensions import Self

    from ConfigSpace import ConfigurationSpace
    from rich.console import RenderableType
    from rich.panel import Panel

    from amltk.evalutors.evaluation_protocol import EvaluationProtocol
    from amltk.optimization.metric import Metric
    from amltk.optimization.trial import Trial
    from amltk.pipeline.components import Choice, Join, Sequential
    from amltk.pipeline.parsers.optuna import OptunaSearchSpace
    from amltk.scheduling import Scheduler

    NodeLike: TypeAlias = (
        set["Node" | "NodeLike"]
        | tuple["Node" | "NodeLike", ...]
        | list["Node" | "NodeLike"]
        | Callable[..., Item]
        | Item
    )

    SklearnPipelineT = TypeVar("SklearnPipelineT", bound=SklearnPipeline)


class OnBeginCallbackSignature(Protocol):
    """A  calllback to further define control flow from
    [`pipeline.optimize()`][amltk.pipeline.node.Node.optimize].

    In one of these callbacks, you can register to specific `@events` of the
    [`Scheduler`][amltk.scheduling.Scheduler] or [`Task`][amltk.scheduling.Task].

    ```python
    pipeline = ...

    # The callback will get the task, scheduler and the history in which results
    # will be stored
    def my_callback(task: Task[..., Trial.Report], scheduler: Scheduler, history: History) -> None:

        # You can do early stopping based on a target metric
        @task.on_result
        def stop_if_target_reached(_: Future, report: Trial.Report) -> None:
            score = report.metrics["accuracy"]
            if score >= 0.95:
                scheduler.stop(stop_msg="Target reached!"))

        # You could also perform early stopping based on iterations
        n = 0
        last_score = 0.0

        @task.on_result
        def stop_if_no_improvement_for_n_runs(_: Future, report: Trial.Report) -> None:
            score = report.metrics["accuracy"]
            if score > last_score:
                n = 0
                last_score = score
            elif n >= 5:
                scheduler.stop()
            else:
                n += 1

        # Really whatever you'd like
        @task.on_result
        def print_if_choice_made(_: Future, report: Trial.Report) -> None:
            if report.config["estimator:__choice__"] == "random_forest":
                print("yay")

        # Every callback will be called here in the main process so it's
        # best not to do anything too heavy here.
        # However you can also submit new tasks or jobs to the scheduler too
        @task.on_result(every=30)  # Do a cleanup sweep every 30 trials
        def keep_on_ten_best_models_on_disk(_: Future, report: Trial.Report) -> None:
            sorted_reports = history.sortby("accuracy")
            reports_to_cleanup = sorted_reports[10:]
            scheduler.submit(some_cleanup_function, reporteds_to_cleanup)

    history = pipeline.optimize(
        ...,
        on_begin=my_callback,
    )
    ```
    """  # noqa: E501

    def __call__(
        self,
        task: Task[[Trial, Node], Trial.Report],
        scheduler: Scheduler,
        history: History,
    ) -> None:
        """Signature for the callback.

        Args:
            task: The task that will be run
            scheduler: The scheduler that will be running the optimization
            history: The history that will be used to collect the results
        """
        ...


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
    """How to display this node in rich."""

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
        """Initialize a choice.

        Args:
            nodes: The nodes that this node leads to
            name: The name of the node
            item: The item attached to this node
            config: The configuration for this node
            space: The search space for this node
            fidelities: The fidelities for this node
            config_transform: A function that transforms the configuration of this node
            meta: Any meta information about this node
        """
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
        """Get the first from [`.nodes`][amltk.pipeline.node.Node.nodes] with `key`."""
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

    def factorize(
        self,
        *,
        min_depth: int = 0,
        max_depth: int | None = None,
        current_depth: int = 0,
        factor_by: Callable[[Node], bool] | None = None,
        assign_child: Callable[[Node, Node], Node] | None = None,
    ) -> Iterator[Self]:
        """Please see [`factorize()`][amltk.pipeline.ops.factorize]."""  # noqa: D402
        from amltk.pipeline.ops import factorize

        yield from factorize(
            self,
            min_depth=min_depth,
            max_depth=max_depth,
            current_depth=current_depth,
            factor_by=factor_by,
            assign_child=assign_child,
        )

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
        from rich.text import Text

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

    def register_optimization_loop(  # noqa: C901, PLR0915, PLR0912
        self,
        target: (
            EvaluationProtocol
            | Callable[[Trial, Node], Trial.Report]
            | Task[[Trial, Node], Trial.Report]
        ),
        metric: Metric | Sequence[Metric],
        *,
        optimizer: type[Optimizer] | Optimizer.CreateSignature | None = None,
        seed: Seed | None = None,
        max_trials: int | None = 3,
        n_workers: int = 1,
        working_dir: str | Path | PathBucket | None = None,
        scheduler: Scheduler | None = None,
        history: History | None = None,
        on_begin: OnBeginCallbackSignature | None = None,
        on_trial_exception: Literal["raise", "end", "continue"] = "raise",
        # Plugin creating arguments
        plugins: Plugin | Iterable[Plugin] | None = None,
        process_memory_limit: int | tuple[int, str] | None = None,
        process_walltime_limit: int | tuple[float, str] | None = None,
        process_cputime_limit: int | tuple[float, str] | None = None,
        threadpool_limit_ctl: bool | int | None = None,
    ) -> Scheduler:
        """Setup a pipeline to be optimized in a loop.

        Args:
            target:
                The function against which to optimize.

                * If `target` is an
                [`EvaluationProtocol`][amltk.evalutors.evaluation_protocol.EvaluationProtocol],
                then it will be used to evaluate the pipeline.

                * If `target` is a function, then it must take in a
                [`Trial`][amltk.optimization.trial.Trial] as the first argument
                and a [`Node`][amltk.pipeline.node.Node] second argument, returning a
                [`Trial.Report`][amltk.optimization.trial.Trial.Report]. Please refer to
                the [optimization guide](../../../guides/optimization.md) for more.

                * If `target` is a [`Task`][amltk.scheduling.task.Task], then
                this is not implemeneted yet. Sorry
            metric:
                The metric(s) that will be passed to `optimizer=`. These metrics
                should align with what is being computed in `target=`.
            optimizer:
                The optimizer to use. If `None`, then AMLTK will go through a list
                of known optimizers and use the first one it can find which was installed.

                Alternatively, this can be a class inheriting from
                [`Optimizer`][amltk.optimization.optimizer.Optimizer] or else
                a signature match [`Optimizer.CreateSignature`][amltk.optimization.Optimizer.CreateSignature]

                ??? tip "`Optimizer.CreateSignature`"

                    ::: amltk.optimization.Optimizer.CreateSignature

            seed:
                A [`seed`][amltk.types.Seed] for the optimizer to use.
            n_workers:
                The numer of workers to use to evaluate this pipeline.
                If no `scheduler=` is provided, then one will be created for
                you as [`Scheduler.with_processes(n_workers)`][amltk.scheduling.Scheduler.with_processes].
                If you provide your own `scheduler=` then this will limit the maximum
                amount of concurrent trials for this pipeline that will be evaluating
                at once.
            working_dir:
                A working directory to use for the optimizer and the trials.
                Any items you store in trials will be located in this directory,
                where the [`trial.name`][amltk.optimization.Trial.name] will be
                used as a subfolder where any contents stored with
                [`trial.store()`][amltk.optimization.trial.Trial.store] will be put there.
                Please see the [optimization guide](../../../guides/optimization.md)
                for more on trial storage.
            scheduler:
                The specific [`Scheduler`][amltk.scheduling.Scheduler] to use.
                If `None`, then one will be created for you with
                [`Scheduler.with_processes(n_workers)`][amltk.scheduling.Scheduler.with_processes]
            history:
                A [`History`][amltk.optimization.history.History] to store the
                [`Trial.Report`][amltk.optimization.Trial.Report]s in. You
                may pass in your own if you wish for this method to store
                it there instead of creating its own.
            on_begin:
                A callback that will be called before the scheduler is run. This
                can be used to hook into the life-cycle of the optimization and
                perform custom routines. Please see the
                [scheduling guide](../../../guides/scheduling.md) for more.

                ??? tip "on_begin signature"

                    ::: amltk.pipeline.node.OnBeginCallbackSignature

            on_trial_exception:
                What to do when a trial returns a fail report from
                [`trial.fail()`][amltk.optimization.trial.Trial.fail] or
                [`trial.crashed()`][amltk.optimization.trial.Trial.crashed]
                that contains an exception.

                Please see the [optimization guide](../../../guides/optimization.md)
                for more. In all cases, the exception will be attached to the
                [`Trial.Report`][amltk.optimization.Trial.Report] object under
                [`report.exception`][amltk.optimization.Trial.Report.exception].

                * If `#!python "raise"`, then the exception will be raised
                immediatly and the optimization process will halt. The default
                and good for initial development.
                * If `#!python "end"`, then the exception will be caught and
                the optimization process will end gracefully.
                * If `#!python "continue"`, the exception will be ignored and
                the optimization procedure will continue.

            max_trials:
                The maximum number of trials to run. If `None`, then the
                optimization will continue for as long as the scheduler is
                running. You'll likely want to configure this.
            process_memory_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the memory the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            process_walltime_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the wall time the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            process_cputime_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the cputime the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            threadpool_limit_ctl:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`ThreadPoolCTLPlugin`][amltk.scheduling.plugins.threadpoolctl.ThreadPoolCTLPlugin]
                to limit the number of threads used by compliant libraries.
                **Notably**, this includes scikit-learn, for which running multiple
                in parallel can be problematic if not adjusted accordingly.

                The default behavior (when `None`) is to auto-detect whether this
                is applicable. This is done by checking if `sklearn` is installed
                and if the first node in the pipeline has a `BaseEstimator` item.
                Please set this to `True`/`False` depending on your preference.
            plugins:
                Additional plugins to attach to the eventual
                [`Task`][amltk.scheduling.task.Task] that will be executed by
                the [`Scheduler`][amltk.scheduling.Scheduler]. Please
                refer to the
                [plugins reference](../../../reference/scheduling/plugins.md) for more.
        """  # noqa: E501
        match history:
            case None:
                history = History()
            case History():
                pass
            case _:
                raise ValueError(f"Invalid history {history}. Must be a History")

        _plugins: tuple[Plugin, ...]
        match plugins:
            case None:
                _plugins = ()
            case Plugin():
                _plugins = (plugins,)
            case Iterable():
                _plugins = tuple(plugins)
            case _:
                raise ValueError(
                    f"Invalid plugins {plugins}. Must be a Plugin or an Iterable of"
                    " Plugins",
                )

        if any(
            limit is not None
            for limit in (
                process_memory_limit,
                process_walltime_limit,
                process_cputime_limit,
            )
        ):
            try:
                from amltk.scheduling.plugins.pynisher import PynisherPlugin
            except ImportError as e:
                raise ImportError(
                    "You must install `pynisher` to use `trial_*_limit`"
                    " You can do so with `pip install amltk[pynisher]`"
                    " or `pip install pynisher` directly",
                ) from e
            # TODO: I'm hesitant to add even more arguments to the `optimize`
            # signature, specifically for `mp_context`.
            plugin = PynisherPlugin(
                memory_limit=process_memory_limit,
                walltime_limit=process_walltime_limit,
                cputime_limit=process_cputime_limit,
            )
            _plugins = (*_plugins, plugin)

        # If threadpool_limit_ctl None, we should default to inspecting if it's
        # an sklearn pipeline. This is because sklearn pipelines
        # run in parallel will over-subscribe the CPU and cause
        # the system to slow down.
        # We use a heuristic to check this by checking if the item at the head
        # of this node is a subclass of sklearn.base.BaseEstimator
        match threadpool_limit_ctl:
            case None:
                from amltk._util import threadpoolctl_heuristic

                threadpool_limit_ctl = False
                if threadpoolctl_heuristic(self.item):
                    threadpool_limit_ctl = 1
                    warnings.warn(
                        "Detected an sklearn pipeline. Setting `threadpool_limit_ctl`"
                        " to True. This will limit the number of threads spawned by"
                        " sklearn to the number of cores on the machine. This is"
                        " because sklearn pipelines run in parallel will over-subscribe"
                        " the CPU and cause the system to slow down."
                        "\nPlease set `threadpool_limit_ctl=False` if you do not want"
                        " this behaviour and set it to `True` to silence this warning.",
                        stacklevel=2,
                    )
            case True:
                threadpool_limit_ctl = 1
            case False:
                pass
            case int():
                pass
            case _:
                raise ValueError(
                    f"Invalid threadpool_limit_ctl {threadpool_limit_ctl}."
                    " Must be a bool or an int",
                )

        if threadpool_limit_ctl is not False:
            from amltk.scheduling.plugins.threadpoolctl import ThreadPoolCTLPlugin

            _plugins = (*_plugins, ThreadPoolCTLPlugin(threadpool_limit_ctl))

        match max_trials:
            case None:
                pass
            case int() if max_trials > 0:
                from amltk.scheduling.plugins import Limiter

                _plugins = (*_plugins, Limiter(max_calls=max_trials))
            case _:
                raise ValueError(f"{max_trials=} must be a positive int")

        from amltk.evalutors.evaluation_protocol import EvaluationProtocol

        match target:
            case EvaluationProtocol():
                pass
            case Task():  # type: ignore # NOTE not sure why pyright complains here
                # TODO: When updating this, please update the docstring too
                raise NotImplementedError(
                    "Not sure how to handle an already created task yet",
                )
            case _ if callable(target):
                from amltk.evalutors.evaluation_protocol import CustomProtocol

                target = CustomProtocol(target)
            case _:
                raise ValueError(
                    f"Invalid target {target}. Must be a function or an"
                    " EvaluationProtocol",
                )

        from amltk.scheduling.scheduler import Scheduler

        match scheduler:
            case None:
                scheduler = Scheduler.with_processes(n_workers)
            case Scheduler():
                pass
            case _:
                raise ValueError(f"Invalid scheduler {scheduler}. Must be a Scheduler")

        # NOTE: I'm not particularly fond of this hack but I assume most people
        # when prototyping don't care for the actual underlying optimizer and
        # so we should just *pick one*.
        create_optimizer: Optimizer.CreateSignature
        match optimizer:
            case None:
                first_opt_class = next(
                    Optimizer._get_known_importable_optimizer_classes(),
                    None,
                )
                if first_opt_class is None:
                    raise ValueError(
                        "No optimizer was given and no known importable optimizers were"
                        " found. Please consider giving one explicitly or  installing"
                        " one of the following packages:\n"
                        "\n - optuna"
                        "\n - smac"
                        "\n - neural-pipeline-search",
                    )

                create_optimizer = first_opt_class.create
                opt_name = classname(first_opt_class)
            case type():
                if not issubclass(optimizer, Optimizer):
                    raise ValueError(
                        f"Invalid optimizer {optimizer}. Must be a subclass of"
                        " Optimizer or a function that returns an Optimizer",
                    )
                create_optimizer = optimizer.create
                opt_name = classname(optimizer)
            case _:
                assert not isinstance(optimizer, type)
                create_optimizer = optimizer
                opt_name = funcname(optimizer)

        match working_dir:
            case None:
                now = datetime.utcnow().isoformat()

                working_dir = PathBucket(f"{opt_name}-{self.name}-{now}")
            case str() | Path():
                working_dir = PathBucket(working_dir)
            case PathBucket():
                pass
            case _:
                raise ValueError(
                    f"Invalid working_dir {working_dir}."
                    " Must be a str, Path or PathBucket",
                )

        _optimizer = create_optimizer(
            space=self,
            metrics=metric,
            bucket=working_dir,
            seed=seed,
        )

        task = target.task(scheduler=scheduler, plugins=_plugins)

        if on_begin is not None:
            hook = partial(on_begin, task, scheduler, history)
            scheduler.on_start(hook)

        @scheduler.on_start
        def launch_initial_trials() -> None:
            trials = _optimizer.ask(n=n_workers)
            for trial in trials:
                task.submit(trial, self)

        from amltk.optimization.trial import Trial

        @task.on_result
        def add_report_to_history(_: Any, report: Trial.Report) -> None:
            history.add(report)
            match report.status:
                case Trial.Status.SUCCESS:
                    return
                case Trial.Status.FAIL | Trial.Status.CRASHED | Trial.Status.UNKNOWN:
                    match on_trial_exception:
                        case "raise":
                            if report.exception is None:
                                raise RuntimeError(
                                    f"Trial finished with status {report.status} but"
                                    " no exception was attached!",
                                )
                            raise report.exception
                        case "end":
                            scheduler.stop(
                                stop_msg=f"Trial finished with status {report.status}",
                                exception=report.exception,
                            )
                        case "continue":
                            pass
                case _:
                    raise ValueError(f"Invalid status {report.status}")

        @task.on_result
        def run_next_trial(*_: Any) -> None:
            if scheduler.running():
                trial = _optimizer.ask()
                task.submit(trial, self)

        return scheduler

    def optimize(
        self,
        target: (
            EvaluationProtocol
            | Callable[[Trial, Node], Trial.Report]
            | Task[[Trial, Node], Trial.Report]
        ),
        metric: Metric | Sequence[Metric],
        *,
        optimizer: type[Optimizer] | Optimizer.CreateSignature | None = None,
        seed: Seed | None = None,
        max_trials: int | None = 3,
        n_workers: int = 1,
        timeout: float | None = None,
        working_dir: str | Path | PathBucket | None = None,
        scheduler: Scheduler | None = None,
        history: History | None = None,
        on_begin: OnBeginCallbackSignature | None = None,
        on_trial_exception: Literal["raise", "end", "continue"] = "raise",
        # Plugin creating arguments
        plugins: Plugin | Iterable[Plugin] | None = None,
        process_memory_limit: int | tuple[int, str] | None = None,
        process_walltime_limit: int | tuple[float, str] | None = None,
        process_cputime_limit: int | tuple[float, str] | None = None,
        threadpool_limit_ctl: bool | int | None = None,
        # `scheduler.run()` arguments
        display: bool | Literal["auto"] = "auto",
        wait: bool = True,
        on_scheduler_exception: Literal["raise", "end", "continue"] = "raise",
    ) -> History:
        """Optimize a pipeline on a given target function or evaluation protocol.

        Args:
            target:
                The function against which to optimize.

                * If `target` is an
                [`EvaluationProtocol`][amltk.evalutors.evaluation_protocol.EvaluationProtocol],
                then it will be used to evaluate the pipeline.

                * If `target` is a function, then it must take in a
                [`Trial`][amltk.optimization.trial.Trial] as the first argument
                and a [`Node`][amltk.pipeline.node.Node] second argument, returning a
                [`Trial.Report`][amltk.optimization.trial.Trial.Report]. Please refer to
                the [optimization guide](../../../guides/optimization.md) for more.

                * If `target` is a [`Task`][amltk.scheduling.task.Task], then
                this is not implemeneted yet. Sorry
            metric:
                The metric(s) that will be passed to `optimizer=`. These metrics
                should align with what is being computed in `target=`.
            optimizer:
                The optimizer to use. If `None`, then AMLTK will go through a list
                of known optimizers and use the first one it can find which was installed.

                Alternatively, this can be a class inheriting from
                [`Optimizer`][amltk.optimization.optimizer.Optimizer] or else
                a signature match [`Optimizer.CreateSignature`][amltk.optimization.Optimizer.CreateSignature]

                ??? tip "`Optimizer.CreateSignature`"

                    ::: amltk.optimization.Optimizer.CreateSignature

            seed:
                A [`seed`][amltk.types.Seed] for the optimizer to use.
            n_workers:
                The numer of workers to use to evaluate this pipeline.
                If no `scheduler=` is provided, then one will be created for
                you as [`Scheduler.with_processes(n_workers)`][amltk.scheduling.Scheduler.with_processes].
                If you provide your own `scheduler=` then this will limit the maximum
                amount of concurrent trials for this pipeline that will be evaluating
                at once.
            timeout:
                How long to run the scheduler for. This parameter only takes
                effect if `setup_only=False` which is the default. Otherwise,
                it will be ignored.
            display:
                Whether to display the scheduler during running. By default
                it is `"auto"` which means to enable the display if running
                in a juptyer notebook or colab. Otherwise, it will be
                `False`.

                This may work poorly if including print statements or logging.
            wait:
                Whether to wait for the scheduler to finish all pending jobs
                if was stopped for any reason, e.g. a `timeout=` or
                [`scheduler.stop()`][amltk.scheduling.Scheduler.stop] was called.
            on_scheduler_exception:
                What to do if an exception occured, either in the submitted task,
                the callback, or any other unknown source during the loop.

                * If `#!python "raise"`, then the exception will be raised
                immediatly and the optimization process will halt. The default
                behavior and good for initial development.
                * If `#!python "end"`, then the exception will be caught and
                the optimization process will end gracefully.
                * If `#!python "continue"`, the exception will be ignored and
                the optimization procedure will continue.
            working_dir:
                A working directory to use for the optimizer and the trials.
                Any items you store in trials will be located in this directory,
                where the [`trial.name`][amltk.optimization.Trial.name] will be
                used as a subfolder where any contents stored with
                [`trial.store()`][amltk.optimization.trial.Trial.store] will be put there.
                Please see the [optimization guide](../../../guides/optimization.md)
                for more on trial storage.
            scheduler:
                The specific [`Scheduler`][amltk.scheduling.Scheduler] to use.
                If `None`, then one will be created for you with
                [`Scheduler.with_processes(n_workers)`][amltk.scheduling.Scheduler.with_processes]
            history:
                A [`History`][amltk.optimization.history.History] to store the
                [`Trial.Report`][amltk.optimization.Trial.Report]s in. You
                may pass in your own if you wish for this method to store
                it there instead of creating its own.
            on_begin:
                A callback that will be called before the scheduler is run. This
                can be used to hook into the life-cycle of the optimization and
                perform custom routines. Please see the
                [scheduling guide](../../../guides/scheduling.md) for more.

                ??? tip "on_begin signature"

                    ::: amltk.pipeline.node.OnBeginCallbackSignature

            on_trial_exception:
                What to do when a trial returns a fail report from
                [`trial.fail()`][amltk.optimization.trial.Trial.fail] or
                [`trial.crashed()`][amltk.optimization.trial.Trial.crashed]
                that contains an exception.

                Please see the [optimization guide](../../../guides/optimization.md)
                for more. In all cases, the exception will be attached to the
                [`Trial.Report`][amltk.optimization.Trial.Report] object under
                [`report.exception`][amltk.optimization.Trial.Report.exception].

                * If `#!python "raise"`, then the exception will be raised
                immediatly and the optimization process will halt. The default
                and good for initial development.
                * If `#!python "end"`, then the exception will be caught and
                the optimization process will end gracefully.
                * If `#!python "continue"`, the exception will be ignored and
                the optimization procedure will continue.

            max_trials:
                The maximum number of trials to run. If `None`, then the
                optimization will continue for as long as the scheduler is
                running. You'll likely want to configure this.

            process_memory_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the memory the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            process_walltime_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the wall time the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            process_cputime_limit:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`PynisherPlugin`][amltk.scheduling.plugins.pynisher.PynisherPlugin]
                to limit the cputime the process can use. Please
                refer to the
                [plugins `pynisher` reference](../../../reference/scheduling/plugins.md#pynisher)
                for more as there are platform limitations and additional
                dependancies required.
            threadpool_limit_ctl:
                If specified, the [`Task`][amltk.scheduling.task.Task] will
                use the
                [`ThreadPoolCTLPlugin`][amltk.scheduling.plugins.threadpoolctl.ThreadPoolCTLPlugin]
                to limit the number of threads used by compliant libraries.
                **Notably**, this includes scikit-learn, for which running multiple
                in parallel can be problematic if not adjusted accordingly.

                The default behavior (when `None`) is to auto-detect whether this
                is applicable. This is done by checking if `sklearn` is installed
                and if the first node in the pipeline has a `BaseEstimator` item.
                Please set this to `True`/`False` depending on your preference.
            plugins:
                Additional plugins to attach to the eventual
                [`Task`][amltk.scheduling.task.Task] that will be executed by
                the [`Scheduler`][amltk.scheduling.Scheduler]. Please
                refer to the
                [plugins reference](../../../reference/scheduling/plugins.md) for more.
        """  # noqa: E501
        match history:
            case None:
                history = History()
            case History():
                pass
            case _:
                raise ValueError(f"Invalid history {history}. Must be a History")

        scheduler = self.register_optimization_loop(
            target=target,
            metric=metric,
            optimizer=optimizer,
            seed=seed,
            max_trials=max_trials,
            n_workers=n_workers,
            working_dir=working_dir,
            scheduler=scheduler,
            history=history,
            on_begin=on_begin,
            on_trial_exception=on_trial_exception,
            plugins=plugins,
            process_memory_limit=process_memory_limit,
            process_walltime_limit=process_walltime_limit,
            process_cputime_limit=process_cputime_limit,
            threadpool_limit_ctl=threadpool_limit_ctl,
        )
        scheduler.run(
            wait=wait,
            timeout=timeout,
            on_exception=on_scheduler_exception,
            display=display,
        )
        return history
