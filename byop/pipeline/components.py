"""The various components that can be part of a pipeline.

These can all be created through the functions `step`, `split`, `choice`
exposed through the `byop.pipeline` module and this is the preffered way to do so.
"""
from __future__ import annotations

from itertools import chain
from typing import Any, Generic, Iterator, Mapping, Sequence, TypeVar

from attrs import field, frozen

from byop.pipeline.step import Key, Step

T = TypeVar("T")
Space = TypeVar("Space")


@frozen(kw_only=True)
class Component(Step[Key], Generic[Key, T]):
    """A Fixed component with an item attached

    Attributes
    ----------
        item: The item attached to this component
        config (optional): Any additional items to associate with this config
    """

    item: T = field(hash=False)
    config: Mapping[str, Any] | None = field(default=None, hash=False)

    def walk(
        self,
        splits: list[Split[Key]] | None = None,
        parents: list[Step[Key]] | None = None,
    ) -> Iterator[tuple[list[Split[Key]] | None, list[Step[Key]] | None, Step[Key]]]:
        """See `Step.walk`"""
        yield splits, parents, self
        parents = parents + [self] if parents is not None else [self]

        if self.nxt is not None:
            yield from self.nxt.walk(splits, parents)

    def traverse(self, *, include_self: bool = True) -> Iterator[Step[Key]]:
        """See `Step.traverse`"""
        yield self
        if self.nxt is not None:
            yield from self.nxt.traverse()


@frozen(kw_only=True)
class Searchable(Component[Key, T], Generic[Key, T, Space]):
    """A Fixed component with an item attached

    Attributes
    ----------
        space: The space to search over
    """

    space: Space = field(hash=False)


@frozen(kw_only=True)
class Split(Step[Key], Generic[Key]):
    """A split in the pipeline

    Attributes
    ----------
        paths: The paths to take
    """

    paths: Sequence[Step[Key]] = field(hash=False)

    def traverse(self, *, include_self: bool = True) -> Iterator[Step[Key]]:
        """See `Step.traverse`"""
        yield self
        yield from chain.from_iterable(path.traverse() for path in self.paths)
        if self.nxt is not None:
            yield from self.nxt.traverse()

    def walk(
        self,
        splits: list[Split[Key]] | None = None,
        parents: list[Step[Key]] | None = None,
    ) -> Iterator[tuple[list[Split[Key]] | None, list[Step[Key]] | None, Step[Key]]]:
        """See `Step.walk`"""
        yield splits, parents, self
        splits = splits + [self] if splits is not None else None
        parents = parents + [self] if parents is not None else [self]

        for path in self.paths:
            yield from path.walk(splits=splits, parents=None)

        if self.nxt:
            yield from self.nxt.walk(splits=splits, parents=parents)


@frozen(kw_only=True)
class Choice(Split[Key]):
    """A Choice between different subcomponents

    Attributes
    ----------
        paths: The choices to choose from
        weights (optional): Any weights to attach to each choice
    """

    weights: Sequence[float] | None = field(hash=False)
