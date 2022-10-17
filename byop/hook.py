from __future__ import annotations

from typing import Callable, Generic, TypeVar

from dataclasses import dataclass, field

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Hook(Generic[P, R]):
    listeners: list[Callable[P, R]] = field(default_factory=list)

    def do(self, f: Callable[P, R]) -> None:
        self.listeners.append(f)

    def emit(self, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        return [listener(*args, **kwargs) for listener in self.listeners]
