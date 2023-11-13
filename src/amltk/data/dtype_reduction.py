"""Reduce the dtypes of data."""
from __future__ import annotations

import logging
from typing import TypeAlias, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataContainer: TypeAlias = np.ndarray | (pd.DataFrame | pd.Series)
D = TypeVar("D", bound=DataContainer)


def reduce_floating_precision(x: D) -> D:
    """Reduce the floating point precision of the data.

    For a float array, will reduce by one step, i.e. float32 -> float16, float64
    -> float32.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_floating_precision(x[col])
        return x  # type: ignore

    if x.dtype.kind != "f":
        return x

    _reduction_map = {
        # Base numpy dtypes
        "float128": "float64",
        "float96": "float64",
        "float64": "float32",
        "float32": "float16",
        # Nullable pandas dtypes (only supports 64 and 32 bit)
        "Float64": "Float32",
    }

    if (dtype := _reduction_map.get(x.dtype.name)) is not None:
        return x.astype(dtype)  # type: ignore

    return x


def reduce_int_span(x: D) -> D:
    """Reduce the integer span of the data.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array.

    Args:
        x: The data to reduce.

    Returns:
        The reduced data.
    """
    # For a dataframe, we recurse over all columns
    if isinstance(x, pd.DataFrame):
        # Using `apply` doesn't work
        for col in x.columns:
            x[col] = reduce_int_span(x[col])
        return x  # type: ignore

    if x.dtype.kind not in "iu":
        return x

    min_dtype = np.min_scalar_type(x.min())  # type: ignore
    max_dtype = np.min_scalar_type(x.max())  # type: ignore
    dtype = np.result_type(min_dtype, max_dtype)
    return x.astype(dtype)  # type: ignore


def reduce_dtypes(x: D, *, reduce_int: bool = True, reduce_float: bool = True) -> D:
    """Reduce the dtypes of data.

    When a dataframe, will reduce the dtypes of all columns.
    When applied to an iterable, will apply to all elements of the iterable.

    For an int array, will reduce to the smallest dtype that can hold the
    minimum and maximum values of the array. Otherwise for floats, will reduce
    by one step, i.e. float32 -> float16, float64 -> float32.

    Args:
        x: The data to reduce.
        reduce_int: Whether to reduce integer dtypes.
        reduce_float: Whether to reduce floating point dtypes.
    """
    if not isinstance(x, pd.DataFrame | pd.Series | np.ndarray):
        raise TypeError(f"Cannot reduce data of type {type(x)}.")

    if isinstance(x, pd.Series | pd.DataFrame):
        x = x.convert_dtypes()

    if reduce_int:
        x = reduce_int_span(x)
    if reduce_float:
        x = reduce_floating_precision(x)

    return x
