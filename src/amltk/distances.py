"""Distance functions.

This module contains functions for calculating the distance between
two vectors.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

_norm = np.linalg.norm

DistanceMetric: TypeAlias = Callable[[npt.ArrayLike, npt.ArrayLike], float]
"""A metric used for calculating distances.

Takes two arrays-like objects and returns a float.
"""


def pnorm(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    p: int | float = 2,
) -> float:
    """Calculates the p-norm between each column in x and y.

    The p-norm is defined as:

        `||x - y||_p = (sum_i(|x_i - y_i|^p))^(1/p)`

    The common values for p are 1, 2 and infinity.

    * [`l1_distance()`][amltk.distances.l1_distance]
    * [`l2_distance()`][amltk.distances.l2_distance]
    * [`linf_distance()`][amltk.distances.linf_distance]

    !!! tip "Using a `partial`"

        To use this function with
        [`dataset_distance()`][amltk.metalearning.dataset_distance],
        you can wrap this in [`functools.partial()`][functools.partial].

        ```python
        from functools import partial
        from amltk.metalearning import dataset_distance
        from amltk.distances import pnorm

        dataset_distance(
            target,
            dataset_metafeatures,
            method=partial(pnorm, p=3), # (1)!
        )
        ```

        1. [`partial()`][functools.partial] creates a new function with the
        `p` argument set to 3.

    Args:
        x: The vector to compare.
        y: The vector to compute the distance to
        p: The p in p-norm.

    Returns:
        A series with the same index as x.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if p is np.inf:
        return float(np.max(np.abs(x - y)))

    return float(np.linalg.norm(x - y, ord=p))


def cosine_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculates the cosine distance between each column in x and y.

    The cosine distance is defined as 1 - cosine_similarity. This means
    the distance is 0 when the vectors are identical, 1 when orthogonal
    and 2 when they are opposite.

    Args:
        x: A dataframe with columns being the features and rows being the samples.
        y: A series with the same index as x.

    Returns:
        A series with the same index as x.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    cosine_similarity = np.dot(x, y) / (_norm(x) * _norm(y))
    return float(1 - cosine_similarity)


l1_distance = partial(pnorm, p=1)
"""Calculates the l1 distance between each column in x and y.

The l1 distance is defined as:

    `||x - y||_1 = sum_i(|x_i - y_i|)`

This is the sum of the absolute differences between each element in x and y.

See Also:
    * [`pnorm()`][amltk.distances.pnorm]
"""

l2_distance = partial(pnorm, p=2)
"""Calculates the l2 distance between each column in x and y.

The l2 distance is defined as:

    `||x - y||_2 = sqrt(sum_i(|x_i - y_i|^2))`

This is the square root of the sum of the squared differences between each
element in x and y.

See Also:
    * [`pnorm()`][amltk.distances.pnorm]
"""

linf_distance = partial(pnorm, p=np.inf)
"""Calculates the linf distance between each column in x and y.

The linf distance is defined as:

    `||x - y||_inf = max_i(|x_i - y_i|)`

This is the maximum absolute difference between each element in x and y.

See Also:
    * [`pnorm()`][amltk.distances.pnorm]
"""

euclidean_distance = l2_distance
"""Calculates the euclidean distance between each column in x and y.

Same as [`l2_distance()`][amltk.distances.l2_distance].
"""

NamedDistance: TypeAlias = Literal["l1", "l2", "euclidean", "cosine", "max"]
"""Predefined distance metrics.

Possible values are:

* `"l1"`: [`l1_distance()`][amltk.distances.l1_distance]
* `"l2"`: [`l2_distance()`][amltk.distances.l2_distance]
* `"euclidean"`: [`euclidean_distance()`][amltk.distances.euclidean_distance]
* `"cosine"`: [`cosine_distance()`][amltk.distances.cosine_distance]
* `"max"`: [`linf_distance()`][amltk.distances.linf_distance]
"""

distance_metrics: dict[NamedDistance, DistanceMetric] = {
    "l1": l1_distance,
    "l2": l2_distance,
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "max": partial(pnorm, p=np.inf),
}


class NearestNeighborsDistance:
    """Uses [sklearn.neighbors.NearestNeighbors][] to calculate the distance."""

    def __init__(self, **nn_kwargs: Any):
        """Creates a new NearestNeighborsDistance.

        Args:
            **nn_kwargs: Keyword arguments to pass to
                [sklearn.neighbors.NearestNeighbors][].
        """
        super().__init__()
        self.nn_kwargs = nn_kwargs

    def __call__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Calculates the distance between each column in x and y.

        Args:
            x: An array-like with columns being the features and rows being the samples.
            y: A array with the same index as x.

        Returns:
            An array with the same index as x.
        """
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(**self.nn_kwargs)

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _y.ndim != 1:
            raise ValueError(f"y must be a 1-dimensional array. Got shape {_y.shape}")

        _y = _y.reshape(1, -1)
        _x = _x.T

        if _x.ndim == 1:
            _x = np.asarray([_x])

        self.nn.fit(_x)
        distances, _ = self.nn.kneighbors(
            _y,
            n_neighbors=len(_x),
            return_distance=True,
        )
        return np.asarray(distances.reshape(-1), dtype=float)
