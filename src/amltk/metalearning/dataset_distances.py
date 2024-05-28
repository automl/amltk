"""Calculating metadata distances."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, TypeVar

import pandas as pd

from amltk._functional import funcname
from amltk.distances import (
    DistanceMetric,
    NamedDistance,
    NearestNeighborsDistance,
    distance_metrics,
)
from amltk.types import safe_isinstance

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


D = TypeVar("D")

DatasetDistanceMetric = Callable[[pd.DataFrame, pd.Series], pd.Series]


def _metric_for_frame(
    _metric: DistanceMetric,
) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    def _new_metric(x: pd.DataFrame, y: pd.Series) -> pd.Series:
        _x = x.apply(lambda col: _metric(col.to_numpy(), y.to_numpy()))
        assert isinstance(_x, pd.Series)
        return _x

    return _new_metric


def dataset_distance(  # noqa: C901, PLR0912
    target: pd.Series,
    dataset_metafeatures: Mapping[str, pd.Series],
    *,
    distance_metric: (DistanceMetric | NearestNeighborsDistance | NamedDistance) = "l2",
    scaler: TransformerMixin
    | Callable[[pd.DataFrame], pd.DataFrame]
    | Literal["minmax"]
    | None = None,
    closest_n: int | None = None,
) -> pd.Series:
    """Calculates the distance between a target dataset and a set of datasets.

    This uses the metafeatures of the datasets to calculate the distance.

    Args:
        target: The target dataset's metafeatures.
        dataset_metafeatures: A dictionary of dataset names to their metafeatures.
        distance_metric: The method to use to calculate the distance.
            Takes in the target dataset's metafeatures and a dataset's metafeatures
            Should return the distance between the two.
        scaler: A scaler to use to scale the metafeatures.
        closest_n: The number of closest datasets to return. If None, all datasets
            are returned.

    Returns:
        Series with the index being the dataset name and the values being the distance.
    """
    outname: str
    if isinstance(distance_metric, str):
        outname = distance_metric
    else:
        outname = funcname(distance_metric)

    if target.name is None:
        target = target.copy()
        target.name = "target-dataset"

    _method = (
        distance_metrics[distance_metric]
        if isinstance(distance_metric, str)
        else distance_metric
    )

    if not isinstance(_method, NearestNeighborsDistance):
        _method = _metric_for_frame(_method)

    metafeatures = {
        name: ds_metafeatures.rename(name)
        for name, ds_metafeatures in dataset_metafeatures.items()
    }

    # Index is dataset name with columns being the values
    #      | mf1 | mf2
    # d1
    # d2
    # d3
    combined = pd.concat([target, *metafeatures.values()], axis=1).T

    if scaler is None:
        pass
    elif scaler == "minmax":
        min_maxs = combined.agg(["min", "max"], axis=0).T

        mins = min_maxs["min"]
        maxs = min_maxs["max"]
        normalizer = maxs - mins
        normalizer[normalizer == 0] = 1
        mins[normalizer == 0] = 0

        norm = lambda col: (col - mins) / normalizer
        combined = combined.apply(norm, axis=1)
    elif safe_isinstance(scaler, "TransformerMixin"):
        combined = scaler.set_output(transform="pandas").fit_transform(  # type: ignore
            combined,
        )
    elif callable(scaler):
        combined = scaler(combined)
    else:
        raise ValueError(f"Unsure how to handle {scaler=}")

    # We now transpose the dataframe so that the index is the metafeature name
    # while the columns are the dataset names
    #   x   | d1 | d2 | d3          y | dy
    #  mf1                      mf1
    #  mf2                      mf2
    x = combined.T.drop(columns=target.name)
    y = combined.loc[target.name]

    # Should return a series with index being dataset names and values being the
    #     | distance
    # d1
    # d2
    dataset_distances = _method(x, y)

    if not isinstance(dataset_distances, pd.Series):
        dataset_distances = pd.Series(
            dataset_distances,
            dtype=float,
            index=list(dataset_metafeatures.keys()),
            name=outname,
        )
    else:
        dataset_distances = dataset_distances.astype(float).rename(outname)

    dataset_distances = dataset_distances.sort_values()

    if closest_n is not None:
        if closest_n > len(dataset_distances):
            warnings.warn(
                f"Cannot get {closest_n} closest datasets when there are"
                f" only {len(dataset_distances)} datasets. Returning all.",
                UserWarning,
                stacklevel=2,
            )

        dataset_distances = dataset_distances.iloc[:closest_n]

    return dataset_distances


if __name__ == "__main__":
    target = pd.Series([1, 1], name="target", index=["mf1", "mf2"])
    dataset_metafeatures = {
        "d1": pd.Series([1, 1], name="d1", index=["mf1", "mf2"]),
        "d2": pd.Series([-1, -1], name="d2", index=["mf1", "mf2"]),
        "d3": pd.Series([1, -1], name="d2", index=["mf1", "mf2"]),
    }
    distances = dataset_distance(
        target,
        dataset_metafeatures,
        distance_metric=NearestNeighborsDistance(algorithm="brute", metric="l1"),
    )

    distances = dataset_distance(
        target,
        dataset_metafeatures,
        distance_metric="cosine",
    )
