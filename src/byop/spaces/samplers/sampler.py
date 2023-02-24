"""The base definition of a Sampler.

It's primary role is to allow sampling from a particular Space.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, overload

from byop.types import Config, Space


class Sampler(ABC, Generic[Space, Config]):
    """A sampler for a search space."""

    @overload
    @classmethod
    @abstractmethod
    def sample(cls, space: Space, n: None = None) -> Config:
        ...

    @overload
    @classmethod
    @abstractmethod
    def sample(cls, space: Space, n: int) -> list[Config]:
        ...

    @classmethod
    @abstractmethod
    def sample(cls, space: Space, n: int | None = None) -> Config | list[Config]:
        """Sample a configuration from the given space."""
        ...

    @classmethod
    @abstractmethod
    def supports(cls, space: Space) -> bool:
        """Check if the space is supported."""
        ...

    @overload
    def __call__(self, space: Space, n: None = None) -> Config:
        ...

    @overload
    def __call__(self, space: Space, n: int) -> list[Config]:
        ...

    def __call__(self, space: Space, n: int | None = None) -> Config | list[Config]:
        """Call the sampler."""
        return self.sample(space, n)

    @classmethod
    def find(cls, space: Space, /) -> type[Sampler[Space, Any]] | None:
        """Find a sampler for the given space."""
        from byop.spaces.samplers import DEFAULT_SAMPLERS

        for sampler in DEFAULT_SAMPLERS:
            if sampler.supports(space):
                return sampler
        return None
