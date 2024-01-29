from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case, parametrize_with_cases

from amltk.optimization.metric import Metric


@dataclass
class MetricTest:
    """A test case for a metric."""

    metric: Metric
    value: float
    expected_loss: float
    expected_distance_from_optimal: float | None
    expected_score: float
    expected_normalized_loss: float
    expected_str: str


@case(tags=["maximize", "bounded"])
def case_metric_score_bounded() -> MetricTest:
    metric = Metric("score_bounded", minimize=False, bounds=(0, 1))
    return MetricTest(
        metric=metric,
        value=0.3,
        expected_loss=-0.3,
        expected_distance_from_optimal=0.7,
        expected_normalized_loss=0.7,
        expected_score=0.3,
        expected_str="score_bounded [0.0, 1.0] (maximize)",
    )


@case(tags=["maximize", "unbounded"])
def case_metric_score_unbounded() -> MetricTest:
    metric = Metric("score_unbounded", minimize=False)
    return MetricTest(
        metric=metric,
        value=0.3,
        expected_loss=-0.3,
        expected_distance_from_optimal=None,
        expected_normalized_loss=-0.3,
        expected_score=0.3,
        expected_str="score_unbounded (maximize)",
    )


@case(tags=["minimize", "unbounded"])
def case_metric_loss_unbounded() -> MetricTest:
    metric = Metric("loss_unbounded", minimize=True)
    return MetricTest(
        metric=metric,
        value=0.8,
        expected_loss=0.8,
        expected_distance_from_optimal=None,
        expected_normalized_loss=0.8,
        expected_score=-0.8,
        expected_str="loss_unbounded (minimize)",
    )


@case(tags=["minimize", "bounded"])
def case_metric_loss_bounded() -> MetricTest:
    metric = Metric("loss_bounded", minimize=True, bounds=(-1, 1))
    return MetricTest(
        metric=metric,
        value=0.8,
        expected_loss=0.8,
        expected_distance_from_optimal=1.8,
        expected_normalized_loss=0.9,
        expected_score=-0.8,
        expected_str="loss_bounded [-1.0, 1.0] (minimize)",
    )


@parametrize_with_cases(argnames="C", cases=".")
def test_metrics_have_expected_outputs(C: MetricTest) -> None:
    assert C.metric.loss(C.value) == C.expected_loss
    if C.expected_distance_from_optimal is not None:
        assert C.metric.distance_to_optimal(C.value) == C.expected_distance_from_optimal
    assert C.metric.score(C.value) == C.expected_score
    assert str(C.metric) == C.expected_str


@parametrize_with_cases(argnames="C", cases=".", has_tag=["maximize"])
def test_metric_value_is_score_if_maximize(C: MetricTest) -> None:
    assert C.value == C.metric.score(C.value)
    assert C.value == -C.metric.loss(C.value)


@parametrize_with_cases(argnames="C", cases=".", has_tag=["minimize"])
def test_metric_value_is_loss_if_minimize(C: MetricTest) -> None:
    assert C.value == C.metric.loss(C.value)
    assert C.value == -C.metric.score(C.value)


@parametrize_with_cases(argnames="C", cases=".")
def test_metric_value_score_is_just_loss_inverted(C: MetricTest) -> None:
    assert C.metric.score(C.value) == -C.metric.loss(C.value)


@parametrize_with_cases(argnames="C", cases=".", has_tag=["minimize", "unbounded"])
def test_minimize_metric_worst_optimal_if_unbounded(C: MetricTest) -> None:
    assert C.metric.worst == float("inf")
    assert C.metric.optimal == float("-inf")


@parametrize_with_cases(argnames="C", cases=".", has_tag=["maximize", "unbounded"])
def test_maximize_metric_worst_optimal_if_unbounded(C: MetricTest) -> None:
    assert C.metric.worst == float("-inf")
    assert C.metric.optimal == float("inf")


@parametrize_with_cases(argnames="C", cases=".", has_tag=["minimize", "bounded"])
def test_minimize_metric_worst_optimal_if_bounded(C: MetricTest) -> None:
    assert C.metric.bounds is not None
    assert C.metric.worst == C.metric.bounds[1]
    assert C.metric.optimal == C.metric.bounds[0]


@parametrize_with_cases(argnames="C", cases=".", has_tag=["maximize", "bounded"])
def test_maximize_metric_worst_optimal_if_bounded(C: MetricTest) -> None:
    assert C.metric.bounds is not None
    assert C.metric.worst == C.metric.bounds[0]
    assert C.metric.optimal == C.metric.bounds[1]


@parametrize_with_cases(argnames="C", cases=".", has_tag=["unbounded"])
def test_distance_to_optimal_is_raises_for_unbounded(C: MetricTest) -> None:
    with pytest.raises(ValueError, match="unbounded"):
        C.metric.distance_to_optimal(C.value)


@parametrize_with_cases(argnames="C", cases=".", has_tag=["bounded"])
def test_distance_to_optimal_is_always_positive_for_bounded(C: MetricTest) -> None:
    assert C.metric.distance_to_optimal(C.value) >= 0


@parametrize_with_cases(argnames="C", cases=".")
def test_normalized_loss(C: MetricTest) -> None:
    assert C.metric.normalized_loss(C.value) == C.expected_normalized_loss


@parametrize_with_cases(argnames="C", cases=".", has_tag=["bounded"])
def test_normalized_loss_for_bounded(C: MetricTest) -> None:
    assert 0 <= C.metric.normalized_loss(C.value) <= 1
    assert C.metric.normalized_loss(C.metric.optimal) == 0
    mid = (C.metric.optimal + C.metric.worst) / 2
    assert C.metric.normalized_loss(mid) == 0.5
    assert C.metric.normalized_loss(C.metric.worst) == 1


@parametrize_with_cases(argnames="C", cases=".", has_tag=["unbounded"])
def test_normalized_loss_for_unbounded_is_loss(C: MetricTest) -> None:
    assert C.metric.normalized_loss(C.value) == C.metric.loss(C.value)


@parametrize_with_cases(argnames="C", cases=".")
def test_metric_serialization(C: MetricTest) -> None:
    s = str(C.metric)
    reconstructed = Metric.from_str(s)
    assert reconstructed == C.metric
