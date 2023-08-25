from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.distances import (
    DistanceMetric,
    NamedDistance,
    NearestNeighborsDistance,
    distance_metrics,
)
from amltk.functional import funcname
from amltk.metalearning.dataset_distances import dataset_distance


@case
@parametrize("metric", distance_metrics.keys())
def case_default_metric(metric: NamedDistance) -> NamedDistance:
    return metric


@case(tags="nn")
def case_nn_metric_l1() -> NearestNeighborsDistance:
    return NearestNeighborsDistance(algorithm="brute", metric="l1")


@case(tags="nn")
def case_nn_metric_l2() -> NearestNeighborsDistance:
    return NearestNeighborsDistance(algorithm="brute", metric="l2")


@case(tags="nn")
def case_nn_ball_tree() -> NearestNeighborsDistance:
    return NearestNeighborsDistance(algorithm="ball_tree")


@case(tags="nn")
def case_nn_auto() -> NearestNeighborsDistance:
    return NearestNeighborsDistance(algorithm="auto")


@parametrize_with_cases("metric", cases=".")
@parametrize(
    "target",
    [
        pd.Series([1, 2, 3], name="target", index=["mf1", "mf2", "mf3"]),
        np.array([1, 2, 3]),
        [1, 2, 3],
    ],
    idgen=lambda target: f"{target.__class__.__name__}",
)
@parametrize(
    "other",
    [
        pd.Series([1, 2, 3], name="other", index=["mf1", "mf2", "mf3"]),
        np.array([1, 2, 3]),
        [1, 2, 3],
    ],
    idgen=lambda other: f"{other.__class__.__name__}",
)
def test_distance_to_itself_is_zero(
    target: npt.ArrayLike,
    other: npt.ArrayLike,
    metric: DistanceMetric | NearestNeighborsDistance,
) -> None:
    target = np.asarray(target)
    target = pd.Series(
        target,
        name="target",
        index=[f"mf{i}" for i in range(len(target))],
    )
    other = np.asarray(other)
    other = pd.Series(
        other,
        name="other",
        index=[f"mf{i}" for i in range(len(other))],
    )
    other2 = other.copy()

    # We use 2 here to make sure the ordering remains correct
    expected = pd.Series([0, 0], index=["other", "other2"], dtype=float)
    distances = dataset_distance(
        target=target,
        dataset_metafeatures={"other": other, "other2": other2},
        distance_metric=metric,
    )

    assert distances.equals(expected)

    if isinstance(metric, str):
        assert distances.name == metric
    else:
        assert distances.name == funcname(metric)
