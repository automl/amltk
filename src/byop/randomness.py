"""Utilities for dealing with randomness."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from byop.types import Seed

MAX_INT = np.iinfo(np.int32).max


def as_rng(seed: Seed | None = None) -> np.random.Generator:
    """Converts a valid seed arg into a numpy.random.Generator instance.

    Args:
        seed: The seed to use

    Returns:
        A valid np.random.Generator object to use
    """
    if isinstance(seed, np.random.Generator):
        return seed

    if isinstance(seed, np.random.RandomState):
        seed = seed.randint(0, MAX_INT)

    if seed is None or isinstance(seed, int):
        return np.random.default_rng(seed)

    raise ValueError(f"Can't use {seed=} to create a numpy.random.Generator instance")


def as_int(seed: Seed | None = None) -> int:
    """Converts a valid seed arg into an integer.

    Args:
        seed: The seed to use

    Returns:
        A valid integer to use as a seed
    """
    if isinstance(seed, int):
        return seed

    if seed is None:
        return np.random.default_rng().integers(0, MAX_INT)

    if isinstance(seed, np.random.Generator):
        return seed.integers(0, MAX_INT)

    if isinstance(seed, np.random.RandomState):
        return seed.randint(0, MAX_INT)

    raise ValueError(f"Can't use {seed=} to create an integer seed")
