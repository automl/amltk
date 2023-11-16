"""Utilities for dealing with randomness."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from amltk.types import Seed

MAX_INT = np.iinfo(np.int32).max
ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def as_rng(seed: Seed | None = None) -> np.random.Generator:
    """Converts a valid seed arg into a numpy.random.Generator instance.

    Args:
        seed: The seed to use

    Returns:
        A valid np.random.Generator object to use
    """
    match seed:
        case None | int() | np.integer():
            return np.random.default_rng(seed)
        case np.random.Generator():
            return seed
        case np.random.RandomState():
            _seed = seed.randint(0, MAX_INT)
            return np.random.default_rng(_seed)

    raise ValueError(f"Can't {seed=} ({type(seed)}) to create numpy.random.Generator")


def as_randomstate(seed: Seed | None = None) -> np.random.RandomState:
    """Converts a valid seed arg into a numpy.random.RandomState instance.

    Args:
        seed: The seed to use

    Returns:
        A valid np.random.RandomSTate object to use
    """
    match seed:
        case None | int() | np.integer():
            return np.random.RandomState(seed)
        case np.random.RandomState():
            return seed
        case np.random.Generator():
            _seed = seed.integers(0, MAX_INT)
            return np.random.RandomState(_seed)

    raise ValueError(f"Can't {seed=} ({type(seed)}) to create numpy.random.RandomState")


def as_int(seed: Seed | None = None) -> int:
    """Converts a valid seed arg into an integer.

    Args:
        seed: The seed to use

    Returns:
        A valid integer to use as a seed
    """
    match seed:
        case None:
            return int(np.random.default_rng().integers(0, MAX_INT))
        case np.integer() | int():
            return int(seed)
        case np.random.Generator():
            return int(seed.integers(0, MAX_INT))
        case np.random.RandomState():
            return int(seed.randint(0, MAX_INT))

    raise ValueError(f"Can't {seed=} ({type(seed)}) to create int")


def randuid(
    k: int = 8,
    *,
    charset: Sequence[str] = ALPHABET,
    seed: Seed | None = None,
) -> str:
    """Generate a random alpha-numeric uuid of a specified length.

    See: https://stackoverflow.com/a/56398787/5332072

    Args:
        k: The length of the uuid to generate
        charset: The charset to use
        seed: The seed to use

    Returns:
        A random uid
    """
    rng = as_rng(seed)
    return "".join(rng.choice(np.asarray(charset), size=k))
