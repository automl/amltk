"""Conversions between different data repesentations and formats."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


# TODO: This is probably much cleaner to just have version for the different
# shapes and classes
def probabilities_to_classes(
    probabilities: npt.NDArray[np.floating],
    classes: np.ndarray,
) -> np.ndarray:
    """Convert probabilities to classes.

    Using code from DummyClassifier `fit` and `predict`
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/dummy.py#L307.

    Args:
        probabilities: The probabilities to convert
        classes: The classes to use

    Returns:
        The classes corresponding to the probabilities
    """
    shape = np.shape(classes)
    if len(shape) == 1:
        n_outputs = 1
        classes = np.asarray([classes])
        probabilities = np.asarray([probabilities])
    elif len(shape) == 2:  # noqa: PLR2004
        n_outputs = len(classes)
    else:
        raise NotImplementedError(f"Don't support `classes` with ndim > 2, {classes}")

    return np.vstack(
        [
            classes[class_index][probabilities[class_index].argmax(axis=1)]
            for class_index in range(n_outputs)
        ],
    ).T


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
    _x = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else x

    if flatten_if_1d and x.ndim == 2 and x.shape[1] == 1:  # noqa: PLR2004
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
