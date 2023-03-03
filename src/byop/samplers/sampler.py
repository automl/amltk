"""The base definition of a Sampler.

It's primary role is to allow sampling from a particular Space.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, overload

from more_itertools import first_true

from byop.types import Config, Seed, Space


class Sampler(ABC, Generic[Space, Config]):
    """A sampler for a search space."""

    @abstractmethod
    def __init__(self, space: Space, *, seed: Seed | None = None) -> None:
        """Initialize the sampler.

        Args:
            space: The space to sample from.
            seed: The seed to use for sampling.
        """

    @overload
    @abstractmethod
    def sample(self, n: None = None) -> Config:
        ...

    @overload
    @abstractmethod
    def sample(self, n: int) -> list[Config]:
        ...

    @abstractmethod
    def sample(self, n: int | None = None) -> Config | list[Config]:
        """Sample a configuration from the given space."""
        ...

    @classmethod
    @abstractmethod
    def supports(cls, space: Any) -> bool:
        """Check if the space is supported."""
        ...

    @overload
    def __call__(self, n: None = None) -> Config:
        ...

    @overload
    def __call__(self, n: int) -> list[Config]:
        ...

    def __call__(self, n: int | None = None) -> Config | list[Config]:
        """Call the sampler."""
        return self.sample(n)

    @classmethod
    def find(cls, space: Space) -> type[Sampler[Space, Config]]:
        """Find a sampler for the given space.

        Args:
            space: The space to find a sampler for.

        Returns:
            The sampler for the given space.
        """
        from byop.samplers import DEFAULT_SAMPLERS

        first_supported = first_true(
            DEFAULT_SAMPLERS,
            pred=lambda s: s.supports(space),
            default=None,
        )

        if first_supported is None:
            raise ValueError(f"No sampler found for {space=} of type {type(space)=}")

        return first_supported
