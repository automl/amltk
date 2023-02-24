"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""
from __future__ import annotations

from typing import Any, Callable, Generic

from byop.optimization.protocols import Optimizer
from byop.spaces import Sampler
from byop.types import Config, Space


class RandomSearch(Optimizer[Config, Any], Generic[Space, Config]):
    """A random search optimizer."""

    def __init__(
        self,
        *,
        space: Space,
        sampler: Callable[[Space], Config] | type[Sampler[Space, Config]] | None = None,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            sampler: The sampler to use to sample from the space.
                If not provided, the sampler will be automatically found.
        """
        self.space = space
        if sampler is None:
            sampler = Sampler.find(space)

        if sampler is None:
            raise ValueError(f"No sampler found for {space=} of type {type(space)=}")

        self._sampler: Callable[[Space], Config]
        if isinstance(sampler, type) and issubclass(sampler, Sampler):
            self._sample = sampler.sample
        else:
            # TODO: No idea why this is saying `sampler` is an `overloaded function`
            self._sample = sampler  # type: ignore

    def ask(self) -> Config:
        """Sample from the space."""
        return self._sample(self.space)

    def tell(self, result: Any) -> None:
        """Do nothing with the result.

        ???+ note
            We do nothing with the results as it's random search
            and does not use the results to do anything useful.

        Args:
            result: The result of the sampled configuration.
        """
