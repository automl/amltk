"""The various components that can be part of a pipeline.

These can all be created through the functions `step`, `split`, `choice`
exposed through the `amltk.pipeline` module and this is the preffered way to do so.
"""
from __future__ import annotations

from contextlib import suppress
from itertools import chain, repeat
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Sequence,
)
from typing_extensions import override

from attrs import field, frozen
from more_itertools import first_true

from amltk.pipeline.step import ParamRequest, Step, mapping_select, prefix_keys
from amltk.types import Config, FidT, Item, Space

if TYPE_CHECKING:
    from rich.console import RenderableType


@frozen(kw_only=True)
class Searchable(Step[Space], Generic[Space]):
    """A step to be searched over.

    See Also:
        [`Step`][amltk.pipeline.step.Step]
    """

    name: str
    """Name of the step"""

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    """The configuration for this step"""

    search_space: Space | None = field(default=None, hash=False, repr=False)
    """The search space for this step"""

    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )
    """The fidelities for this step"""

    config_transform: (
        Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ]
        | None
    ) = field(default=None, hash=False, repr=False)
    """A function that transforms the configuration of this step"""

    meta: Mapping[str, Any] | None = None
    """Any meta information about this step"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "light_steel_blue"


@frozen(kw_only=True)
class Component(Step[Space], Generic[Item, Space]):
    """A Fixed component with an item attached.

    See Also:
        [`Step`][amltk.pipeline.step.Step]
    """

    item: Callable[..., Item] | Any = field(hash=False)
    """The item attached to this step"""

    name: str
    """Name of the step"""

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    """The configuration for this step"""

    search_space: Space | None = field(default=None, hash=False, repr=False)
    """The search space for this step"""

    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )
    """The fidelities for this step"""

    config_transform: (
        Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ]
        | None
    ) = field(default=None, hash=False, repr=False)
    """A function that transforms the configuration of this step"""

    meta: Mapping[str, Any] | None = None
    """Any meta information about this step"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "default"

    @override
    def build(self, **kwargs: Any) -> Item:
        """Build the item attached to this component.

        Args:
            **kwargs: Any additional arguments to pass to the item

        Returns:
            Item
                The built item
        """
        if callable(self.item):
            config = self.config or {}
            return self.item(**{**config, **kwargs})

        if self.config is not None:
            raise ValueError(f"Can't pass config to a non-callable item in step {self}")

        return self.item

    @override
    def _rich_table_items(self) -> Iterator[tuple[RenderableType, ...]]:
        from rich.pretty import Pretty

        from amltk.richutil import Function

        if self.item is not None:
            if callable(self.item):
                yield "item", Function(self.item)
            else:
                yield "item", Pretty(self.item)

        yield from super()._rich_table_items()


@frozen(kw_only=True)
class Group(Mapping[str, Step], Step[Space]):
    """A Fixed component with an item attached.

    See Also:
        [`Step`][amltk.pipeline.step.Step]
    """

    paths: Sequence[Step]
    """The paths that can be taken from this split"""

    name: str
    """Name of the step"""

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    """The configuration for this step"""

    search_space: Space | None = field(default=None, hash=False, repr=False)
    """The search space for this step"""

    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )
    """The fidelities for this step"""

    config_transform: (
        Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ]
        | None
    ) = field(default=None, hash=False, repr=False)
    """A function that transforms the configuration of this step"""

    meta: Mapping[str, Any] | None = None
    """Any meta information about this step"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "deep_sky_blue2"

    def __attrs_post_init__(self) -> None:
        """Ensure that the paths are all unique."""
        if len(self) != len(set(self)):
            raise ValueError("Paths must be unique")

        for path_head in self.paths:
            object.__setattr__(path_head, "parent", self)
            object.__setattr__(path_head, "prv", None)

    @override
    def path_to(
        self,
        key: str | Step | Callable[[Step], bool],
        *,
        direction: Literal["forward", "backward"] | None = None,
    ) -> list[Step] | None:
        """See [`Step.path_to`][amltk.pipeline.step.Step.path_to]."""
        if callable(key):
            pred = key
        elif isinstance(key, Step):
            pred = lambda step: step == key
        else:
            pred = lambda step: step.name == key

        # We found our target, just return now
        if pred(self):
            return [self]

        if direction in (None, "forward"):
            for member in self.paths:
                if path := member.path_to(pred, direction="forward"):
                    return [self, *path]

            if self.nxt is not None and (
                path := self.nxt.path_to(pred, direction="forward")
            ):
                return [self, *path]

        if direction in (None, "backward"):
            back = self.prv or self.parent
            if back and (path := back.path_to(pred, direction="backward")):
                return [self, *path]

            return None

        return None

    @override
    def traverse(
        self,
        *,
        include_self: bool = True,
        backwards: bool = False,
    ) -> Iterator[Step]:
        """See `Step.traverse`."""
        if include_self:
            yield self

        # Backward mode
        if backwards:
            if self.prv is not None:
                yield from self.prv.traverse(backwards=True)
            elif self.parent is not None:
                yield from self.parent.traverse(backwards=True)

            if include_self:
                yield self

            return

        # Forward mode
        yield from chain.from_iterable(path.traverse() for path in self.paths)
        if self.nxt is not None:
            yield from self.nxt.traverse()

    @override
    def walk(
        self,
        groups: Sequence[Group] | None = None,
        parents: Sequence[Step] | None = None,
    ) -> Iterator[tuple[list[Group], list[Step], Step]]:
        """See `Step.walk`."""
        groups = list(groups) if groups is not None else []
        parents = list(parents) if parents is not None else []
        yield groups, parents, self

        for path in self.paths:
            yield from path.walk(groups=[*groups, self], parents=[])

        if self.nxt:
            yield from self.nxt.walk(
                groups=groups,
                parents=[*parents, self],
            )

    @override
    def replace(self, replacements: Mapping[str, Step]) -> Iterator[Step]:
        """See `Step.replace`."""
        if self.name in replacements:
            yield replacements[self.name]
        else:
            # Otherwise, we need to call replace over any paths and create a new
            # split with those replacements
            paths = [
                Step.join(path.replace(replacements=replacements))
                for path in self.paths
            ]
            yield self.mutate(paths=paths)

        if self.nxt is not None:
            yield from self.nxt.replace(replacements=replacements)

    @override
    def remove(self, keys: Sequence[str]) -> Iterator[Step]:
        """See `Step.remove`."""
        if self.name not in keys:
            # We need to call remove on all the paths. If this removes a
            # path that only has one entry, leading to an empty path, then
            # we ignore any errors from joining and remove the path
            paths = []
            for path in self.paths:
                with suppress(ValueError):
                    new_path = Step.join(path.remove(keys))
                    paths.append(new_path)

            yield self.mutate(paths=paths)

        if self.nxt is not None:
            yield from self.nxt.remove(keys)

    @override
    def apply(self, func: Callable[[Step], Step]) -> Step:
        """Apply a function to all the steps in this group.

        Args:
            func: The function to apply

        Returns:
            Step: The new group
        """
        new_paths = [path.apply(func) for path in self.paths]
        new_nxt = self.nxt.apply(func) if self.nxt else None

        # NOTE: We can't be sure that the function will return a new instance of
        # `self` so we have to make a copy of `self` and then apply the function
        # to that copy.
        new_self = func(self.copy())

        if new_nxt is not None:
            # HACK: Frozen objects do not allow setting attributes after
            # instantiation. Join the two steps together.
            object.__setattr__(new_self, "nxt", new_nxt)
            object.__setattr__(new_self, "paths", new_paths)
            object.__setattr__(new_nxt, "prv", new_self)

        return new_self

    # OPTIMIZE: Unlikely to be an issue but I figure `.items()` on
    # a split of size `n` will cause `n` iterations of `paths`
    # Fixable by implementing more of the `Mapping` functions

    @override
    def __getitem__(self, key: str) -> Step:
        if val := first_true(self.paths, pred=lambda p: p.name == key):
            return val
        raise KeyError(key)

    @override
    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(p.name for p in self.paths)

    @override
    def configure(  # noqa: PLR0912, C901
        self,
        config: Config,
        *,
        prefixed_name: bool | None = None,
        transform_context: Any | None = None,
        params: Mapping[str, Any] | None = None,
        clear_space: bool | Literal["auto"] = "auto",
    ) -> Step:
        """Configure this step and anything following it with the given config.

        Args:
            config: The configuration to apply
            prefixed_name: Whether items in the config are prefixed by the names
                of the steps.
                * If `None`, the default, then `prefixed_name` will be assumed to
                    be `True` if this step has a next step or if the config has
                    keys that begin with this steps name.
                * If `True`, then the config will be searched for items prefixed
                    by the name of the step (and subsequent chained steps).
                * If `False`, then the config will be searched for items without
                    the prefix, i.e. the config keys are exactly those matching
                    this steps search space.
            transform_context: Any context to give to `config_transform=` of individual
                steps.
            params: The params to match any requests when configuring this step.
                These will match against any ParamRequests in the config and will
                be used to fill in any missing values.
            clear_space: Whether to clear the search space after configuring.
                If `"auto"` (default), then the search space will be cleared of any
                keys that are in the config, if the search space is a `dict`. Otherwise,
                `True` indicates that it will be removed in the returned step and
                `False` indicates that it will remain as is.

        Returns:
            Step: The configured step
        """
        if prefixed_name is None:
            if any(key.startswith(f"{self.name}:") for key in config):
                prefixed_name = True
            else:
                prefixed_name = self.nxt is not None

        nxt = (
            self.nxt.configure(
                config,
                prefixed_name=prefixed_name,
                transform_context=transform_context,
                params=params,
                clear_space=clear_space,
            )
            if self.nxt
            else None
        )

        # Configure all the paths, we assume all of these must
        # have the prefixed name and hence use `mapping_select`
        subconfig = mapping_select(config, f"{self.name}:") if prefixed_name else config

        paths = [
            path.configure(
                subconfig,
                prefixed_name=True,
                transform_context=transform_context,
                params=params,
                clear_space=clear_space,
            )
            for path in self.paths
        ]

        this_config = subconfig if prefixed_name else config

        # The config for this step is anything that doesn't have
        # another delimiter in it and is not a part of a subpath
        this_config = {
            k: v
            for k, v in this_config.items()
            if ":" not in k and not any(k.startswith(f"{p.name}") for p in self.paths)
        }

        if self.config is not None:
            this_config = {**self.config, **this_config}

        _params = params or {}
        reqs = [(k, v) for k, v in this_config.items() if isinstance(v, ParamRequest)]
        for k, request in reqs:
            if request.key in _params:
                this_config[k] = _params[request.key]
            elif request.has_default:
                this_config[k] = request.default
            elif request.required:
                raise ParamRequest.RequestNotMetError(
                    f"Missing required parameter {request.key} for step {self.name}"
                    " and no default was provided."
                    f"\nThe request given was: {request}",
                    f"Please use the `params=` argument to provide a value for this"
                    f" request. What was given was `{params=}`",
                )

        # If we have a `dict` for a space, then we can remove any configured keys that
        # overlap it.
        _space: Any
        if clear_space == "auto":
            _space = self.search_space
            if isinstance(self.search_space, dict) and any(self.search_space):
                _overlap = set(this_config).intersection(self.search_space)
                _space = {
                    k: v for k, v in self.search_space.items() if k not in _overlap
                }
                if len(_space) == 0:
                    _space = None

        elif clear_space is True:
            _space = None
        else:
            _space = self.search_space

        if self.config_transform is not None:
            this_config = self.config_transform(this_config, transform_context)

        new_self = self.mutate(
            paths=paths,
            config=this_config if this_config else None,
            nxt=nxt,
            search_space=_space,
        )

        if nxt is not None:
            # HACK: This is a hack to to modify the fact `nxt` is a frozen
            # object. Frozen objects do not allow setting attributes after
            # instantiation.
            object.__setattr__(nxt, "prv", new_self)

        return new_self

    def first(self) -> Step:
        """Get the first step in this group."""
        return self.paths[0]

    @override
    def select(self, choices: Mapping[str, str]) -> Iterator[Step]:
        """See `Step.select`."""
        if self.name in choices:
            choice = choices[self.name]
            chosen = first_true(self.paths, pred=lambda path: path.name == choice)
            if chosen is None:
                raise ValueError(
                    f"{self.__class__.__qualname__} {self.name} has no path '{choice}'"
                    f"\n{self}",
                )
            yield chosen
        else:
            # Otherwise, we need to call select over the paths
            paths = [Step.join(path.select(choices)) for path in self.paths]
            yield self.mutate(paths=paths)

        if self.nxt is not None:
            yield from self.nxt.select(choices)

    @override
    def fidelities(self) -> dict[str, FidT]:
        """See `Step.fidelities`."""
        fids = {}
        for path in self.paths:
            fids.update(prefix_keys(path.fidelities(), f"{self.name}:"))

        if self.nxt is not None:
            fids.update(self.nxt.fidelities())

        return fids

    @override
    def linearized_fidelity(self, value: float) -> dict[str, int | float | Any]:
        """Get the linearized fidelity for this step.

        Args:
            value: The value to linearize. Must be between [0, 1]

        Return:
            dictionary from key to it's linearized fidelity.
        """
        assert 1.0 <= value <= 100.0, f"{value=} not in [1.0, 100.0]"  # noqa: PLR2004
        d = {}
        if self.fidelity_space is not None:
            for f_name, f_range in self.fidelity_space.items():
                low, high = f_range
                fid = low + (high - low) * value
                fid = low + (high - low) * (value - 1) / 100
                fid = fid if isinstance(low, float) else round(fid)
                d[f_name] = fid

            d = prefix_keys(d, f"{self.name}:")

        for path in self.paths:
            path_fids = prefix_keys(path.linearized_fidelity(value), f"{self.name}:")
            d.update(path_fids)

        if self.nxt is None:
            return d

        nxt_fids = self.nxt.linearized_fidelity(value)
        return {**d, **nxt_fids}

    @override
    def _rich_panel_contents(self) -> Iterator[RenderableType]:
        from rich.console import Group as RichGroup
        from rich.table import Table
        from rich.text import Text

        if panel_contents := list(self._rich_table_items()):
            table = Table.grid(padding=(0, 1), expand=False)
            for tup in panel_contents:
                table.add_row(*tup, style="default")
            table.add_section()

            yield table

        if any(self.paths):
            # HACK : Unless we exposed this through another function, we
            # just assume this is desired behaviour.
            connecter = Text("↓", style="bold", justify="center")

            pipeline_table = Table.grid(padding=(0, 1), expand=False)
            pipelines = [RichGroup(*p._rich_iter(connecter)) for p in self.paths]
            pipeline_table.add_row(*pipelines)

            yield pipeline_table


@frozen(kw_only=True)
class Split(Group[Space], Generic[Item, Space]):
    """A split in the pipeline.

    See Also:
        * [`Step`][amltk.pipeline.step.Step]
        * [`Group`][amltk.pipeline.components.Group]
    """

    item: Callable[..., Item] | Any | None = field(default=None, hash=False)
    """The item attached to this step"""

    paths: Sequence[Step]
    """The paths that can be taken from this split"""

    name: str
    """Name of the step"""

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    """The configuration for this step"""

    search_space: Space | None = field(default=None, hash=False, repr=False)
    """The search space for this step"""

    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )
    """The fidelities for this step"""

    config_transform: (
        Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ]
        | None
    ) = field(default=None, hash=False, repr=False)
    """A function that transforms the configuration of this step"""

    meta: Mapping[str, Any] | None = None
    """Any meta information about this step"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "chartreuse4"

    @override
    def build(self, **kwargs: Any) -> Item:
        """Build the item attached to this component.

        Args:
            **kwargs: Any additional arguments to pass to the item

        Returns:
            Item
                The built item
        """
        if self.item is None:
            raise ValueError(f"Can't build a split without an item in step {self}")

        if callable(self.item):
            config = self.config or {}
            return self.item(**{**config, **kwargs})

        if self.config is not None:
            raise ValueError(f"Can't pass config to a non-callable item in step {self}")

        return self.item

    @override
    def _rich_table_items(self) -> Iterator[tuple[RenderableType, ...]]:
        from rich.pretty import Pretty

        from amltk.richutil import Function

        if self.item is not None:
            if callable(self.item):
                yield "item", Function(self.item)
            else:
                yield "item", Pretty(self.item)

        yield from super()._rich_table_items()


@frozen(kw_only=True)
class Choice(Group[Space]):
    """A Choice between different subcomponents.

    See Also:
        * [`Step`][amltk.pipeline.step.Step]
        * [`Group`][amltk.pipeline.components.Group]
    """

    paths: Sequence[Step]
    """The paths that can be taken from this choice"""

    weights: Sequence[float] | None = field(hash=False)
    """The weights to assign to each path"""

    name: str
    """Name of the step"""

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    """The configuration for this step"""

    search_space: Space | None = field(default=None, hash=False, repr=False)
    """The search space for this step"""

    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )
    """The fidelities for this step"""

    config_transform: (
        Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ]
        | None
    ) = field(default=None, hash=False, repr=False)
    """A function that transforms the configuration of this step"""

    meta: Mapping[str, Any] | None = None
    """Any meta information about this step"""

    RICH_PANEL_BORDER_COLOR: ClassVar[str] = "orange4"

    def iter_weights(self) -> Iterator[tuple[Step, float]]:
        """Iter over the paths with their weights."""
        return zip(self.paths, (repeat(1) if self.weights is None else self.weights))

    @override
    def configure(  # noqa: PLR0912, C901
        self,
        config: Config,
        *,
        prefixed_name: bool | None = None,
        transform_context: Any | None = None,
        params: Mapping[str, Any] | None = None,
        clear_space: bool | Literal["auto"] = "auto",
    ) -> Step:
        """Configure this step and anything following it with the given config.

        Args:
            config: The configuration to apply
            prefixed_name: Whether items in the config are prefixed by the names
                of the steps.
                * If `None`, the default, then `prefixed_name` will be assumed to
                    be `True` if this step has a next step or if the config has
                    keys that begin with this steps name.
                * If `True`, then the config will be searched for items prefixed
                    by the name of the step (and subsequent chained steps).
                * If `False`, then the config will be searched for items without
                    the prefix, i.e. the config keys are exactly those matching
                    this steps search space.
            transform_context: The context to pass to the config transform function.
            params: The params to match any requests when configuring this step.
                These will match against any ParamRequests in the config and will
                be used to fill in any missing values.
            clear_space: Whether to clear the search space after configuring.
                If `"auto"` (default), then the search space will be cleared of any
                keys that are in the config, if the search space is a `dict`. Otherwise,
                `True` indicates that it will be removed in the returned step and
                `False` indicates that it will remain as is.

        Returns:
            Step: The configured step
        """
        if prefixed_name is None:
            if any(key.startswith(self.name) for key in config):
                prefixed_name = True
            else:
                prefixed_name = self.nxt is not None

        nxt = (
            self.nxt.configure(
                config,
                prefixed_name=prefixed_name,
                transform_context=transform_context,
                params=params,
                clear_space=clear_space,
            )
            if self.nxt
            else None
        )

        # For a choice to be made, the config must have the a key
        # for the name of this choice and the choice made.
        chosen_path_name = config.get(self.name)

        if chosen_path_name is not None:
            chosen_path = first_true(
                self.paths,
                pred=lambda path: path.name == chosen_path_name,
            )
            if chosen_path is None:
                raise Step.ConfigurationError(
                    step=self,
                    config=config,
                    reason=f"Choice {self.name} has no path '{chosen_path_name}'",
                )

            # Configure the chosen path
            subconfig = mapping_select(config, f"{self.name}:")
            chosen_path = chosen_path.configure(
                subconfig,
                prefixed_name=prefixed_name,
                transform_context=transform_context,
                params=params,
                clear_space=clear_space,
            )

            object.__setattr__(chosen_path, "old_parent", self.name)

            if nxt is not None:
                # HACK: This is a hack to to modify the fact `nxt` is a frozen
                # object. Frozen objects do not allow setting attributes after
                # instantiation.
                object.__setattr__(nxt, "prv", chosen_path)
                object.__setattr__(chosen_path, "nxt", nxt)

            return chosen_path

        # Otherwise there is no chosen path and we simply configure what we can
        # of the choices and return that
        subconfig = mapping_select(config, f"{self.name}:")
        paths = [
            path.configure(
                subconfig,
                prefixed_name=True,
                transform_context=transform_context,
                params=params,
                clear_space=clear_space,
            )
            for path in self.paths
        ]

        # The config for this step is anything that doesn't have
        # another delimiter in it
        config_for_this_choice = {k: v for k, v in subconfig.items() if ":" not in k}

        if self.config is not None:
            config_for_this_choice = {**self.config, **config_for_this_choice}

        _params = params or {}
        reqs = [
            (k, v)
            for k, v in config_for_this_choice.items()
            if isinstance(v, ParamRequest)
        ]
        for k, request in reqs:
            if request.key in _params:
                config_for_this_choice[k] = _params[request.key]
            elif request.has_default:
                config_for_this_choice[k] = request.default
            elif request.required:
                raise ParamRequest.RequestNotMetError(
                    f"Missing required parameter {request.key} for step {self.name}"
                    " and no default was provided."
                    f"\nThe request given was: {request}",
                    f"Please use the `params=` argument to provide a value for this"
                    f" request. What was given was `{params=}`",
                )

        # If we have a `dict` for a space, then we can remove any configured keys that
        # overlap it.
        _space: Any
        if clear_space == "auto":
            _space = self.search_space
            if isinstance(self.search_space, dict) and any(self.search_space):
                _overlap = set(config_for_this_choice).intersection(self.search_space)
                _space = {
                    k: v for k, v in self.search_space.items() if k not in _overlap
                }
                if len(_space) == 0:
                    _space = None

        elif clear_space is True:
            _space = None
        else:
            _space = self.search_space

        if self.config_transform is not None:
            _config_for_this_choice = self.config_transform(
                config_for_this_choice,
                transform_context,
            )
        else:
            _config_for_this_choice = config_for_this_choice

        new_self = self.mutate(
            paths=paths,
            config=_config_for_this_choice if _config_for_this_choice else None,
            nxt=nxt,
            search_space=_space,
        )

        if nxt is not None:
            # HACK: This is a hack to to modify the fact `nxt` is a frozen
            # object. Frozen objects do not allow setting attributes after
            # instantiation.
            object.__setattr__(nxt, "prv", new_self)

        return new_self
