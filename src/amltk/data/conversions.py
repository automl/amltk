"""Conversions between different data repesentations and formats."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def probabilities_to_classes(
    probabilities: npt.NDArray[np.floating],
    classes: np.ndarray | npt.ArrayLike | list,
) -> np.ndarray:
    """Convert probabilities to classes.

    !!! note

        Converts using the logic of `predict()` of `RandomForestClassifier`.

    Args:
        probabilities: The probabilities to convert
        classes: The classes to use.

    Returns:
        The classes corresponding to the probabilities
    """
    # Taken from `predict()` of RandomForestclassifier
    classes = np.asarray(classes)
    n_outputs = 1 if classes.ndim == 1 else classes.shape[1]
    if n_outputs == 1:
        return classes.take(np.argmax(probabilities, axis=1), axis=0)  # type: ignore

    n_samples = probabilities[0].shape[0]
    # all dtypes should be the same, so just take the first
    class_type = classes[0].dtype
    predictions = np.empty((n_samples, n_outputs), dtype=class_type)

    for k in range(n_outputs):
        predictions[:, k] = classes[k].take(np.argmax(probabilities[k], axis=1), axis=0)

    return predictions


def to_numpy(
    x: np.ndarray | pd.DataFrame | pd.Series,
    *,
    flatten_if_1d: bool = False,
) -> np.ndarray:
    """Convert to numpy array.

    Args:
        x: The data to convert
        flatten_if_1d: Whether to flatten the array if it is 1d

    Returns:
        The converted data
    """
    _x = x.to_numpy() if isinstance(x, pd.DataFrame | pd.Series) else np.asarray(x)

    if (
        flatten_if_1d
        and x.ndim == 2  # noqa: PLR2004 # type: ignore
        and x.shape[1] == 1  # type: ignore
    ):
        _x = np.ravel(_x)

    assert isinstance(_x, np.ndarray)
    return _x


@overload
def flatten_if_1d(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    ...


@overload
def flatten_if_1d(x: np.ndarray) -> np.ndarray:  # type: ignore
    ...


def flatten_if_1d(
    x: np.ndarray | pd.DataFrame | pd.Series,
) -> np.ndarray | pd.DataFrame | pd.Series:
    """Flatten if 1d.

    Retains the type of the input, i.e. pandas stays pandas and numpy stays numpy.

    Args:
        x: The data to flatten

    Returns:
        The flattened data
    """
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 1:  # noqa: PLR2004
        x = np.ravel(x)
    elif (
        isinstance(x, pd.DataFrame) and x.ndim == 2 and x.shape[1] == 1  # noqa: PLR2004
    ):
        x = x.iloc[:, 0]

    return x


def is_str_object_dtype(x: np.ndarray) -> bool:
    """Check if object dtype and string values.

    Args:
        x: The data to check

    Returns:
        Whether it is object dtype and string values
    """
    return x.dtype == object and isinstance(x[0], str)


def as_str_dtype_if_str_object(x: np.ndarray) -> np.ndarray:
    """Convert to string dtype if object dtype and string values.

    Args:
        x: The data to convert

    Returns:
        The converted data if it can be done
    """
    if is_str_object_dtype(x):
        return x.astype(str)
    return x
