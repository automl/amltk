"""A Configspace sampler."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from byop.randomness import as_rng
from byop.samplers.sampler import Sampler
from byop.types import Seed, safe_isinstance

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace


class ConfigSpaceSampler(Sampler["ConfigurationSpace"]):
    """A sampler for a search space."""

    def __init__(
        self, space: ConfigurationSpace, /, *, seed: Seed | None = None
    ) -> None:
        """Initialize the sampler."""
        self.space = space
        self.rng = as_rng(seed)
        int_seed = self.rng.integers(0, 2**32 - 1)
        space.seed(int_seed)

    @overload
    def sample(self, n: None = None) -> Configuration:
        ...

    @overload
    def sample(self, n: int) -> list[Configuration]:
        ...

    def sample(self, n: int | None = None) -> Configuration | list[Configuration]:
        """Sample a configuration from the given space."""
        if n is None:
            return self.space.sample_configuration()
        if n == 1:
            return [self.space.sample_configuration()]

        return self.space.sample_configuration(n)

    @classmethod
    def supports(cls, space: Any) -> bool:
        """Check if the space is a ConfigurationSpace."""
        return safe_isinstance(space, "ConfigurationSpace")
