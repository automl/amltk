"""Conversions between different data repesentations and formats."""
from __future__ import annotations

import numpy as np
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
        classes = [classes]  # type: ignore
        probabilities = [probabilities]  # type: ignore
    elif len(shape) == 2:  # noqa: PLR2004
        n_outputs = len(classes)
    else:
        raise NotImplementedError(f"Don't support `classes` with ndim > 2, {classes}")

    return np.vstack(
        [
            classes[class_index][probabilities[class_index].argmax(axis=1)]
            for class_index in range(n_outputs)
        ]
    ).T
