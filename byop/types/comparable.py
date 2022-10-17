from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, TypeVar

CT = TypeVar("CT", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __lt__(self: CT, other: CT | Any) -> bool:
        ...
