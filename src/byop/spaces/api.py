"""TODO:."""
from __future__ import annotations

from typing import Any, Callable, Literal, overload

from more_itertools import first_true

from byop.spaces.samplers import DEFAULT_SAMPLERS, Sampler
from byop.types import Config, Space


@overload
def sample(space: Any, *, sampler: Literal["auto"] = "auto", n: None = None) -> Any:
    ...


@overload
def sample(space: Any, *, sampler: Literal["auto"] = "auto", n: int) -> list[Any]:
    ...


@overload
def sample(
    space: Space,
    *,
    sampler: Callable[[Space], Config] | Sampler[Space, Config],
    n: None = None,
) -> Config:
    ...


@overload
def sample(
    space: Space,
    *,
    sampler: Callable[[Space], Config] | Sampler[Space, Config],
    n: int,
) -> list[Config]:
    ...


def sample(
    space: Space,
    *,
    sampler: (
        Literal["auto"] | Callable[[Space], Config] | Sampler[Space, Config]
    ) = "auto",
    n: int | None = None,
) -> Config | list[Config]:
    """Sample a configuration from the given space."""
    if sampler == "auto":
        valid_sampler = first_true(DEFAULT_SAMPLERS, pred=lambda s: s.supports(space))
        if valid_sampler is None:
            raise ValueError(f"No sampler found for space in {DEFAULT_SAMPLERS}")
        return valid_sampler.sample(space, n=n)

    if isinstance(sampler, Sampler):
        return sampler.sample(space, n=n)

    if n is None:
        return sampler(space)

    return [sampler(space) for _ in range(n)]
