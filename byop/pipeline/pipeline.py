from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

from collections import deque
from dataclasses import dataclass

from more_itertools import all_unique, first_true
from typing_extensions import TypeAlias

Self = TypeVar("Self", bound="Pipeline")
Config = TypeVar("Config")
Space = TypeVar("Space")
Item = TypeVar("Item")


@dataclass
class Component(Generic[Item]):
    name: str
    item: Callable[..., Item] | Item
    kwargs: Mapping[str, Any]
    inject: Sequence[str] | None


@dataclass
class Configurable(Component[Item], Generic[Item, Space]):
    name: str
    space: Space


@dataclass
class Choice:
    name: str
    choices: Sequence[Node]
    weights: Sequence[float] | None

    def __post_init__(self) -> None:
        if len(self.choices) <= 1:
            raise ValueError("Must provide at least two choices")

        if self.weights is not None and len(self.weights) != len(self.choices):
            raise ValueError(
                "Must provide one weight for each choices if providing weights"
            )


Node: TypeAlias = Union[Component, Configurable, Choice]


class Pipeline(Generic[Space, Config]):
    def __init__(self, *steps: Node):
        self.steps = steps

        if not all_unique(self.iter(mode="pre"), key=lambda s: s.name):
            raise ValueError(f"Duplicate step names in the pipeline, {self.steps}")

    def __contains__(self, key: str) -> bool:
        return self.find(key) is not None

    @overload
    def __getitem__(self, key: str | int) -> Node:
        ...

    @overload
    def __getitem__(self: Self, key: slice | Sequence[int]) -> Self:
        ...

    # Note: Overload str vs Sequence[Str]
    #
    #  We would like to include Sequence[str] into the overload above but according to
    #  mypy this doesn't work because techinically str and Sequence[str] are the same
    #  https://github.com/python/typing/issues/256
    #
    def __getitem__(
        self: Self,
        key: str | int | slice | Sequence[str] | Sequence[int],
    ) -> Node | Self:
        if isinstance(key, str):
            if step := self.find(key):
                return step
            raise KeyError(key)

        if isinstance(key, int):
            return self.steps[key]

        if isinstance(key, slice):
            return self.steps[key]  # type: ignore

        if isinstance(key, Sequence):
            cls = self.__class__
            if len(key) == 0:
                raise KeyError(f"List of keys must not be empty, got {key}")
            if isinstance(key[0], int):
                key = cast(Sequence[int], key)
                return cls(*[self.steps[i] for i in key])
            if isinstance(key[0], str):
                key = cast(Sequence[str], key)
                return cls(*[step for k in key if (step := self.find(k)) is not None])

        raise KeyError(f"Unknown key type {key}")

    def subpipe(self: Self, keys: Iterable[str]) -> Self:
        steps = [self[k] for k in keys]
        return self.__class__(*steps)

    def find(self, name: str) -> Node | None:
        return first_true(
            self.iter("pre"),
            pred=lambda node: node is not None and node.name == name,
            default=None,
        )

    def iter(
        self,
        mode: Literal["pre", "post", "top"] = "top",
        only_leaves: bool = False,
    ) -> Iterator[Node]:
        if mode == "pre":
            queue = deque(self.steps)
            while len(queue) > 0:
                step = queue.popleft()
                if isinstance(step, Choice):
                    queue.extendleft(step.choices)

                    if not only_leaves:
                        yield step
                else:
                    yield step

        elif mode == "top":
            yield from iter(self.steps)

        else:
            raise NotImplementedError(mode)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Node]:
        return self.iter(mode="top")

    def build(
        self,
        config: Config,
        **kwargs: Any,
    ) -> dict[str, Any]:
        selection = self.select(config)
        components: dict = {}
        for (node, node_config), step in zip(selection, self):
            conf = {**node_config, **node.kwargs}
            if kwargs is not None and node.inject:
                conf.update(
                    {item for k in node.inject if (item := kwargs.get(k)) if not None}
                )
            components[step.name] = node.item(**conf)

        return components

    @abstractmethod
    def select(
        self, config: Config
    ) -> list[tuple[Component | Configurable, dict[str, Any]]]:
        ...

    @abstractmethod
    def space(self) -> Config:
        ...
