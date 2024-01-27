from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from more_itertools import all_unique
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.optimization import Metric, Optimizer, Trial
from amltk.pipeline import Component
from amltk.profiling import Timer

if TYPE_CHECKING:
    from amltk.optimization.optimizers.neps import NEPSOptimizer
    from amltk.optimization.optimizers.optuna import OptunaOptimizer
    from amltk.optimization.optimizers.smac import SMACOptimizer

logger = logging.getLogger(__name__)


class _A:
    pass


metrics = [
    Metric("score_bounded", minimize=False, bounds=(0, 1)),
    Metric("score_unbounded", minimize=False),
    Metric("loss_unbounded", minimize=True),
    Metric("loss_bounded", minimize=True, bounds=(-1, 1)),
]


def target_function(trial: Trial, err: Exception | None = None) -> Trial.Report:
    """A target function for testing optimizers."""
    with trial.profile("trial"):
        # Do stuff with trail.info here
        logger.debug(trial.info)

        if err is not None:
            return trial.fail(err)

        return trial.success(
            **{metric.name: metric.optimal.value for metric in trial.metrics},
        )


def valid_time_interval(interval: Timer.Interval) -> bool:
    """Check if the start and end time are valid."""
    return interval.start <= interval.end


@parametrize("metric", [*metrics, metrics])  # Single obj and multi
def opt_smac_hpo(metric: Metric, tmp_path: Path) -> SMACOptimizer:
    try:
        from amltk.optimization.optimizers.smac import SMACOptimizer
    except ImportError:
        pytest.skip("SMAC is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1.0, 10.0)})
    return SMACOptimizer.create(
        space=pipeline,
        bucket=tmp_path,
        metrics=metric,
        seed=42,
    )


@case
@parametrize("metric", [*metrics, metrics])  # Single obj and multi
def opt_optuna(metric: Metric, tmp_path: Path) -> OptunaOptimizer:
    try:
        from amltk.optimization.optimizers.optuna import OptunaOptimizer
    except ImportError:
        pytest.skip("Optuna is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return OptunaOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=42,
        bucket=tmp_path,
    )


@case
@parametrize("metric", [*metrics])  # Single obj
def opt_neps(metric: Metric, tmp_path: Path) -> NEPSOptimizer:
    try:
        from amltk.optimization.optimizers.neps import NEPSOptimizer
    except ImportError:
        pytest.skip("NEPS is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return NEPSOptimizer.create(
        space=pipeline,
        metrics=metric,
        overwrite=True,
        bucket=tmp_path,
        working_dir=tmp_path,
    )


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_report_success(optimizer: Optimizer) -> None:
    """Test that the optimizer can report a success."""
    trial = optimizer.ask()
    report = target_function(trial, err=None)
    optimizer.tell(report)

    assert report.status == Trial.Status.SUCCESS
    assert valid_time_interval(report.time)
    assert report.trial.info is trial.info
    assert report.metric_values == tuple(metric.optimal for metric in optimizer.metrics)


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_report_failure(optimizer: Optimizer):
    trial = optimizer.ask()
    report = target_function(trial, err=ValueError("Error inside Target Function"))
    optimizer.tell(report)
    assert report.status is Trial.Status.FAIL

    assert valid_time_interval(report.time)
    assert isinstance(report.exception, ValueError)
    assert isinstance(report.traceback, str)
    assert report.metric_values == tuple(metric.worst for metric in optimizer.metrics)


@parametrize_with_cases("optimizer", cases=".", prefix="opt_")
def test_batched_ask_generates_unique_configs(optimizer: Optimizer):
    """Test that batched ask generates unique configs."""
    # NOTE: This was tested with up to 100, at least from SMAC and Optuna.
    # It was quite slow for smac so I've reduced it to 10.
    # This is not a hard requirement of optimizers (maybe it should be?)
    batch = list(optimizer.ask(10))
    assert len(batch) == 10
    assert all_unique(batch)
