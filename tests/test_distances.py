from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.distances import DistanceMetric, NearestNeighborsDistance, distance_metrics


@case
@parametrize("metric", distance_metrics.values())
def case_default_metric(metric: DistanceMetric) -> DistanceMetric:
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
    metric: DistanceMetric,
) -> None:
    assert metric(target, other) == 0
