"""The various components that can be part of a pipeline.

These can all be created through the functions `step`, `split`, `choice`
exposed through the `byop.pipeline` module and this is the preffered way to do so.
"""
from __future__ import annotations

from contextlib import suppress
from itertools import chain, repeat
from typing import Any, Callable, Generic, Iterator, Mapping, Sequence

from attrs import field, frozen
from more_itertools import first_true

from byop.pipeline.step import Step
from byop.types import Item, Space


@frozen(kw_only=True)
class Component(Step, Generic[Item, Space]):
    """A Fixed component with an item attached.

    Attributes:
        name: The name of the component
        item: The item attached to this component
        config (optional): Any additional items to associate with this config
        space (optional): A search space associated with this component
    """

    name: str
    item: Callable[..., Item] | Item = field(hash=False)

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    space: Space | None = field(default=None, hash=False, repr=False)

    def walk(
        self,
        splits: Sequence[Split],
        parents: Sequence[Step],
    ) -> Iterator[tuple[list[Split], list[Step], Step]]:
        """See `Step.walk`."""
        splits = list(splits)
        parents = list(parents)
        yield splits, parents, self

        if self.nxt is not None:
            yield from self.nxt.walk(splits, [*parents, self])

    def traverse(self, *, include_self: bool = True) -> Iterator[Step]:
        """See `Step.traverse`."""
        if include_self:
            yield self

        if self.nxt is not None:
            yield from self.nxt.traverse()  # type: ignore

    def replace(self, replacements: Mapping[str, Step]) -> Iterator[Step]:
        """See `Step.replace`."""
        yield replacements.get(self.name, self)

        if self.nxt is not None:
            yield from self.nxt.replace(replacements=replacements)

    def remove(self, keys: Sequence[str]) -> Iterator[Step]:
        """See `Step.remove`."""
        if self.name not in keys:
            yield self

        if self.nxt is not None:
            yield from self.nxt.remove(keys)

    def select(self, choices: Mapping[str, str]) -> Iterator[Step]:
        """See `Step.select`."""
        yield self

        if self.nxt is not None:
            yield from self.nxt.select(choices)

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
class Split(Mapping[str, Step], Step, Generic[Item, Space]):
    """A split in the pipeline.

    Attributes:
        name: The name of the component
        paths: The paths that can be taken from this split
        item (optional): The item attached to this component
        config (optional): Any additional items to associate with this config
        space (optional): A search space associated with this component
    """

    name: str
    paths: Sequence[Step] = field(hash=False)

    item: Item | Callable[..., Item] | None = field(default=None, hash=False)
    config: Mapping[str, Any] | None = field(default=None, hash=False)
    space: Space | None = field(default=None, hash=False, repr=False)

    def traverse(self, *, include_self: bool = True) -> Iterator[Step]:
        """See `Step.traverse`."""
        if include_self:
            yield self

        yield from chain.from_iterable(path.traverse() for path in self.paths)
        yield from self.nxt.traverse() if self.nxt else []

    def walk(
        self,
        splits: Sequence[Split],
        parents: Sequence[Step],
    ) -> Iterator[tuple[list[Split], list[Step], Step]]:
        """See `Step.walk`."""
        splits = list(splits)
        parents = list(parents)
        yield splits, parents, self

        for path in self.paths:
            yield from path.walk(splits=[*splits, self], parents=[])

        if self.nxt:
            yield from self.nxt.walk(
                splits=splits,
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

    def select(self, choices: Mapping[str, str]) -> Iterator[Step]:
        """See `Step.select`."""
        yield self

        if self.nxt is not None:
            yield from self.nxt.select(choices)

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
class Choice(Split[Item, Space]):
    """A Choice between different subcomponents.

    Attributes:
        name: The name of the component
        paths: The paths that can be taken from this split
        weights (optional): The weights associated with each path
        item (optional): The item attached to this component
        config (optional): Any additional items to associate with this config
        space (optional): A search space associated with this component
    """

    name: str
    paths: Sequence[Step] = field(hash=False)

    weights: Sequence[float] | None = field(hash=False)

    item: Item | Callable[..., Item] | None = field(default=None, hash=False)
    config: Mapping[str, Any] | None = field(default=None, hash=False)
    space: Space | None = field(default=None, hash=False, repr=False)

    def iter_weights(self) -> Iterator[tuple[Step, float]]:
        """Iter over the paths with their weights."""
        return zip(self.paths, (repeat(1) if self.weights is None else self.weights))

    def select(self, choices: Mapping[str, str]) -> Iterator[Step]:
        """See `Step.select`."""
        if self.name in choices:
            choice = choices[self.name]
            chosen = first_true(self.paths, pred=lambda path: path.name == choice)
            if chosen is None:
                raise ValueError(f"Choice {self.name} has no path '{choice}'\n{self}")
            yield chosen
        else:
            # Otherwise, we need to call select over the paths
            paths = [Step.join(path.select(choices)) for path in self.paths]
            yield self.mutate(paths=paths)

        if self.nxt is not None:
            yield from self.nxt.select(choices)
