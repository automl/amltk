"""A Configspace sampler."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

from byop.spaces.samplers.sampler import Sampler
from byop.types import safe_isinstance

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace


class ConfigSpaceSampler(Sampler["ConfigurationSpace", "Configuration"]):
    """A sampler for a search space."""

    @overload
    @classmethod
    def sample(cls, space: ConfigurationSpace, n: None = None) -> Configuration:
        ...

    @overload
    @classmethod
    def sample(cls, space: ConfigurationSpace, n: int) -> list[Configuration]:
        ...

    @classmethod
    def sample(
        cls,
        space: ConfigurationSpace,
        n: int | None = None,
    ) -> Configuration | list[Configuration]:
        """Sample a configuration from the given space."""
        if n is None:
            return space.sample_configuration()
        if n == 1:
            return [space.sample_configuration()]

        return space.sample_configuration(n)

    @classmethod
    def supports(cls, space: ConfigurationSpace) -> bool:
        """Check if the space is a ConfigurationSpace."""
        return safe_isinstance(space, "ConfigurationSpace")
