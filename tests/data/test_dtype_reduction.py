from __future__ import annotations

import numpy as np
import pandas as pd
from pytest_cases import parametrize

from amltk.data.dtype_reduction import reduce_dtypes


def test_reduce_dtypes_mixed_df() -> None:
    # Default 8 bytes per number
    mixed_df = pd.DataFrame({"a": np.arange(100), "b": np.linspace(1, 100, 100)})
    reduced_df = reduce_dtypes(mixed_df)

    assert reduced_df["a"].dtype == np.uint8
    assert reduced_df["b"].dtype == np.float32


@parametrize(
    "dtype, expected",
    [
        # For int's we squeeze to smallest possible that holds, max/min
        (np.uint8, np.uint8),
        (np.uint16, np.uint8),
        (np.uint32, np.uint8),
        (np.uint64, np.uint8),
        (np.int8, np.uint8),
        (np.int16, np.uint8),
        (np.int32, np.uint8),
        (np.int64, np.uint8),
        # For floats, we only do single step in precision reduction
        (np.float16, np.float16),
        (np.float32, np.float16),
        (np.float64, np.float32),
    ],
)
def test_reduce_dtypes_series(dtype: np.dtype, expected: np.dtype) -> None:
    series = pd.Series(np.arange(100), dtype=dtype)
    reduced_series = reduce_dtypes(series)
    assert reduced_series.dtype == expected


@parametrize(
    "dtype, expected",
    [
        # For int's we squeeze to smallest possible that holds, max/min
        (np.uint8, np.uint8),
        (np.uint16, np.uint8),
        (np.uint32, np.uint8),
        (np.uint64, np.uint8),
        (np.int8, np.uint8),
        (np.int16, np.uint8),
        (np.int32, np.uint8),
        (np.int64, np.uint8),
        # For floats, we only do single step in precision reduction
        (np.float16, np.float16),
        (np.float32, np.float16),
        (np.float64, np.float32),
    ],
)
def test_reduce_dtypes_np(dtype: np.dtype, expected: np.dtype) -> None:
    series = np.arange(100, dtype=dtype)
    reduced_series = reduce_dtypes(series)
    assert reduced_series.dtype == expected
