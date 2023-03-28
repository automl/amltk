"""API for sampling configurations from a search space."""
from __future__ import annotations

from typing import overload

from byop.samplers.configspace import ConfigSpaceSampler
from byop.samplers.sampler import Sampler
from byop.types import Config, Seed, Space

DEFAULT_SAMPLERS: list[type[Sampler]] = [ConfigSpaceSampler]


@overload
def sample(
    space: Space,
    *,
    sampler: Sampler[Space] | None = ...,
    n: None = None,
    seed: Seed | None = ...,
) -> Config:
    ...


@overload
def sample(
    space: Space,
    *,
    sampler: Sampler[Space] | None = ...,
    n: int,
    seed: Seed | None = ...,
) -> list[Config]:
    ...


def sample(
    space: Space,
    *,
    sampler: Sampler[Space] | None = None,
    n: int | None = None,
    seed: Seed | None = None,
) -> Config | list[Config]:
    """Sample a configuration from the given space."""
    if sampler is None:
        for sampler_cls in DEFAULT_SAMPLERS:
            if sampler_cls.supports(space):
                sampler = sampler_cls(space, seed=seed)
                break
        else:
            raise ValueError(f"No sampler found for space {type(space)=}")

    return sampler.sample(n)
