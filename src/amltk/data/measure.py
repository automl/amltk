"""Measure things about data."""
from __future__ import annotations

from typing import Iterable, Union
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd

DataContainer: TypeAlias = Union[np.ndarray, pd.DataFrame, pd.Series]


def byte_size(data: DataContainer | Iterable[DataContainer]) -> float:
    """Measure the size of data.

    Works for numpy-arrays, pandas DataFrames and Series, and iterables of any of
    these.

    Args:
        data: The data to measure.

    Returns:
        The size of the data.
    """
    if isinstance(data, np.ndarray):
        return data.nbytes
    if isinstance(data, pd.DataFrame):
        return float(data.memory_usage(deep=True).sum())
    if isinstance(data, pd.Series):
        return float(data.memory_usage(deep=True))
    if isinstance(data, Iterable):
        return sum(byte_size(d) for d in data)

    raise TypeError(f"Cannot measure data of type {type(data)}.")
