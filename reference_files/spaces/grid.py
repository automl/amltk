from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence, TypeVar

from attrs import define
from typing_extensions import TypeAlias

from byop.pipeline.components import Configurable, FixedComponent, Searchable
from byop.pipeline.step import Key

T = TypeVar("T")
Components: TypeAlias = FixedComponent | Configurable | Searchable[T, Mapping]


@define
class Grid(Mapping[Key, Sequence[Any]]):
    dimensions: Mapping[Key, Sequence[Any]]

    def __getitem__(self, __k: Key) -> Sequence[Any]:
        pass

    def __len__(self) -> int:
        return len(self.dimensions)

    def __iter__(self) -> Iterator[Key]:
        return iter(self.dimensions)

    def sample(self) -> Mapping[Key, Any]:
        ...
