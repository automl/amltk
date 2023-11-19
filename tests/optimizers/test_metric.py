from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case, parametrize_with_cases

from amltk.optimization.metric import Metric


@dataclass
class MetricTest:
    """A test case for a metric."""

    metric: Metric
    v: Metric.Value
    expected_loss: float
    expected_distance_from_optimal: float | None
    expected_score: float
    expected_str: str


@case(tags=["maximize", "bounded"])
def case_metric_score_bounded() -> MetricTest:
    metric = Metric("score_bounded", minimize=False, bounds=(0, 1))
    return MetricTest(
        metric=metric,
        v=metric(0.3),
        expected_loss=-0.3,
        expected_distance_from_optimal=0.7,
        expected_score=0.3,
        expected_str="score_bounded [0.0, 1.0] (maximize)",
    )


@case(tags=["maximize", "unbounded"])
def case_metric_score_unbounded() -> MetricTest:
    metric = Metric("score_unbounded", minimize=False)
    return MetricTest(
        metric=metric,
        v=metric(0.3),
        expected_loss=-0.3,
        expected_distance_from_optimal=None,
        expected_score=0.3,
        expected_str="score_unbounded (maximize)",
    )


@case(tags=["minimize", "unbounded"])
def case_metric_loss_unbounded() -> MetricTest:
    metric = Metric("loss_unbounded", minimize=True)
    return MetricTest(
        metric=metric,
        v=metric(0.8),
        expected_loss=0.8,
        expected_distance_from_optimal=None,
        expected_score=-0.8,
        expected_str="loss_unbounded (minimize)",
    )


@case(tags=["minimize", "bounded"])
def case_metric_loss_bounded() -> MetricTest:
    metric = Metric("loss_bounded", minimize=True, bounds=(-1, 1))
    return MetricTest(
        metric=metric,
        v=metric(0.8),
        expected_loss=0.8,
        expected_distance_from_optimal=1.8,
        expected_score=-0.8,
        expected_str="loss_bounded [-1.0, 1.0] (minimize)",
    )


@parametrize_with_cases(argnames="C", cases=".")
def test_metrics_have_expected_outputs(C: MetricTest) -> None:
    assert C.v.loss == C.expected_loss
    assert C.v.distance_to_optimal == C.expected_distance_from_optimal
    assert C.v.score == C.expected_score
    assert str(C.metric) == C.expected_str


@parametrize_with_cases(argnames="C", cases=".", has_tag=["maximize"])
def test_metric_value_is_score_if_maximize(C: MetricTest) -> None:
    assert C.v.value == C.v.score
    assert C.v.value == -C.v.loss


@parametrize_with_cases(argnames="C", cases=".", has_tag=["minimize"])
def test_metric_value_is_loss_if_minimize(C: MetricTest) -> None:
    assert C.v.value == C.v.loss
    assert C.v.value == -C.v.score


@parametrize_with_cases(argnames="C", cases=".")
def test_metric_value_score_is_just_loss_inverted(C: MetricTest) -> None:
    assert C.v.score == -C.v.loss


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
def test_distance_to_optimal_is_none_for_unbounded(C: MetricTest) -> None:
    assert C.v.distance_to_optimal is None


@parametrize_with_cases(argnames="C", cases=".", has_tag=["bounded"])
def test_distance_to_optimal_is_always_positive_for_bounded(C: MetricTest) -> None:
    assert C.v.distance_to_optimal
    assert C.v.distance_to_optimal >= 0


@parametrize_with_cases(argnames="C", cases=".")
def test_metric_serialization(C: MetricTest) -> None:
    s = str(C.metric)
    reconstructed = Metric.from_str(s)
    assert reconstructed == C.metric
