"""The various components that can be part of a pipeline.

These can all be created through the functions `step`, `split`, `choice`
exposed through the `byop.pipeline` module and this is the preffered way to do so.
"""
from __future__ import annotations

from contextlib import suppress
from itertools import chain, repeat
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Sequence,
)

from attrs import field, frozen
from more_itertools import first_true

from byop.pipeline.step import Step, mapping_select, prefix_keys
from byop.types import Config, FidT, Item, Space


@frozen(kw_only=True)
class Component(Step[Space], Generic[Item, Space]):
    """A Fixed component with an item attached.

    Attributes:
        name: The name of the component
        item: The item attached to this component
        config (optional): Any additional items to associate with this config
        search_space (optional): A search space associated with this component
        fidelity_space: The fidelities for this step
    """

    name: str
    item: Callable[..., Item] | Item = field(hash=False)

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    search_space: Space | None = field(default=None, hash=False, repr=False)
    fidelity_space: Mapping[str, FidT] | None = field(
        default=None,
        hash=False,
        repr=False,
    )

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


@frozen(kw_only=True)
class Group(Mapping[str, Step], Step[Space]):
    """A Fixed component with an item attached.

    Attributes:
        name: The name of the group
        paths: The different paths in the group
    """

    name: str
    paths: Sequence[Step]

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    search_space: Space | None = field(default=None, hash=False, repr=False)
    fidelity_space: Mapping[str, Any] | None = field(
        default=None,
        hash=False,
        repr=False,
    )

    def __attrs_post_init__(self) -> None:
        """Ensure that the paths are all unique."""
        if len(self) != len(set(self)):
            raise ValueError("Paths must be unique")

        for path_head in self.paths:
            object.__setattr__(path_head, "parent", self)
            object.__setattr__(path_head, "prv", None)

    def path_to(
        self,
        key: str | Step | Callable[[Step], bool],
        *,
        direction: Literal["forward", "backward"] | None = None,
    ) -> list[Step] | None:
        """See [`Step.path_to`][byop.pipeline.step.Step.path_to]."""
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

    # OPTIMIZE: Unlikely to be an issue but I figure `.items()` on
    # a split of size `n` will cause `n` iterations of `paths`
    # Fixable by implementing more of the `Mapping` functions

    def __getitem__(self, key: str) -> Step:
        if val := first_true(self.paths, pred=lambda p: p.name == key):
            return val
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[str]:
        return iter(p.name for p in self.paths)

    def configure(self, config: Config, *, prefixed_name: bool | None = None) -> Step:
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

        Returns:
            Step: The configured step
        """
        if prefixed_name is None:
            if any(key.startswith(f"{self.name}:") for key in config):
                prefixed_name = True
            else:
                prefixed_name = self.nxt is not None

        nxt = (
            self.nxt.configure(config, prefixed_name=prefixed_name)
            if self.nxt
            else None
        )

        # Configure all the paths, we assume all of these must
        # have the prefixed name and hence use `mapping_select`
        subconfig = mapping_select(config, f"{self.name}:") if prefixed_name else config

        paths = [path.configure(subconfig, prefixed_name=True) for path in self.paths]

        # The config for this step is anything that doesn't have
        # another delimiter in it
        this_config = subconfig if prefixed_name else config
        this_config = {k: v for k, v in this_config.items() if ":" not in k}

        if self.config is not None:
            this_config = {**self.config, **this_config}

        new_self = self.mutate(
            paths=paths,
            config=this_config if this_config else None,
            nxt=nxt,
            search_space=None,
        )

        if nxt is not None:
            # HACK: This is a hack to to modify the fact `nxt` is a frozen
            # object. Frozen objects do not allow setting attributes after
            # instantiation.
            object.__setattr__(nxt, "prv", new_self)

        return new_self

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

    def fidelities(self) -> dict[str, FidT]:
        """See `Step.fidelities`."""
        fids = {}
        for path in self.paths:
            fids.update(prefix_keys(path.fidelities(), f"{self.name}:"))

        if self.nxt is not None:
            fids.update(self.nxt.fidelities())

        return fids

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


@frozen(kw_only=True)
class Split(Group[Space], Generic[Item, Space]):
    """A split in the pipeline.

    Attributes:
        name: The name of the component
        paths: The paths that can be taken from this split
        item (optional): The item attached to this component
        config (optional): Any additional items to associate with this config
        search_space (optional): A search space associated with this component
    """

    name: str
    paths: Sequence[Step] = field(hash=False)

    item: Item | Callable[..., Item] | None = field(default=None, hash=False)
    config: Mapping[str, Any] | None = field(default=None, hash=False)
    search_space: Space | None = field(default=None, hash=False, repr=False)

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


@frozen(kw_only=True)
class Choice(Group[Space]):
    """A Choice between different subcomponents.

    Attributes:
        name: The name of the component
        paths: The paths that can be taken from this split
        weights: The weights associated with each path
        config (optional): Any additional items to associate with this config
        search_space (optional): A search space associated with this component
    """

    name: str
    paths: Sequence[Step] = field(hash=False)

    weights: Sequence[float] | None = field(hash=False)

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    search_space: Space | None = field(default=None, hash=False, repr=False)

    def iter_weights(self) -> Iterator[tuple[Step, float]]:
        """Iter over the paths with their weights."""
        return zip(self.paths, (repeat(1) if self.weights is None else self.weights))

    def configure(self, config: Config, *, prefixed_name: bool | None = None) -> Step:
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

        Returns:
            Step: The configured step
        """
        if prefixed_name is None:
            if any(key.startswith(self.name) for key in config):
                prefixed_name = True
            else:
                prefixed_name = self.nxt is not None

        nxt = (
            self.nxt.configure(config, prefixed_name=prefixed_name)
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
            chosen_path = chosen_path.configure(subconfig, prefixed_name=prefixed_name)

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
        paths = [path.configure(subconfig, prefixed_name=True) for path in self.paths]

        # The config for this step is anything that doesn't have
        # another delimiter in it
        config_for_this_choice = {k: v for k, v in subconfig.items() if ":" not in k}

        if self.config is not None:
            config_for_this_choice = {**self.config, **config_for_this_choice}

        new_self = self.mutate(
            paths=paths,
            config=config_for_this_choice if config_for_this_choice else None,
            nxt=nxt,
            search_space=None,
        )

        if nxt is not None:
            # HACK: This is a hack to to modify the fact `nxt` is a frozen
            # object. Frozen objects do not allow setting attributes after
            # instantiation.
            object.__setattr__(nxt, "prv", new_self)

        return new_self
