"""Utilities for dealing with randomness."""
from __future__ import annotations

import numpy as np

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


def as_rng(
    seed: int | np.random.RandomState | np.random.Generator | None = None,
) -> np.random.Generator:
    """Converts a valid seed arg into a numpy.random.Generator instance.

    Args:
        seed: The seed to use

    Returns:
        A valid np.random.Generator object to use
    """
    if isinstance(seed, np.random.Generator):
        return seed

    if isinstance(seed, np.random.RandomState):
        seed = seed.randint(MIN_INT, MAX_INT)

    if seed is None or isinstance(seed, int):
        return np.random.default_rng()

    raise ValueError(f"Can't use {seed=} to create a numpy.random.Generator instance")
