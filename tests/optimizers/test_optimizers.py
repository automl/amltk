from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.optimization import Optimizer, Trial
from amltk.pipeline import Component
from amltk.profiling import Memory, Timer

if TYPE_CHECKING:
    from amltk.optimization.optimizers.neps import NEPSOptimizer
    from amltk.optimization.optimizers.optuna import OptunaOptimizer
    from amltk.optimization.optimizers.smac import SMACOptimizer

logger = logging.getLogger(__name__)


class _A:
    pass


def target_function(
    trial: Trial,
    /,
    time_kind: Timer.Kind,
    mem_unit: Memory.Unit,
    key_to_report_in: str,
    err: Exception | None = None,
) -> Trial.Report:
    """A target function for testing optimizers."""
    with trial.begin(time=time_kind, memory_unit=mem_unit):
        # Do stuff with trail.info here
        logger.debug(trial.info)

        if err is not None:
            raise err

        return trial.success(**{key_to_report_in: 1})

    return trial.fail(**{key_to_report_in: 2000})  # pyright: ignore


def valid_time_interval(interval: Timer.Interval) -> bool:
    """Check if the start and end time are valid."""
    return interval.start <= interval.end


@case
def opt_smac_hpo() -> tuple[SMACOptimizer, str]:
    try:
        from amltk.optimization.optimizers.smac import SMACOptimizer
    except ImportError:
        pytest.skip("SMAC is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    return SMACOptimizer.create(
        space=pipeline.search_space(SMACOptimizer.preferred_parser()),
        seed=2**32 - 1,
    ), "cost"


@case
def opt_optuna() -> tuple[OptunaOptimizer, str]:
    try:
        from amltk.optimization.optimizers.optuna import OptunaOptimizer
    except ImportError:
        pytest.skip("Optuna is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    space = pipeline.search_space(parser=OptunaOptimizer.preferred_parser())
    return OptunaOptimizer.create(space=space), "cost"


@case
def opt_neps() -> tuple[NEPSOptimizer, str]:
    try:
        from amltk.optimization.optimizers.neps import NEPSOptimizer
    except ImportError:
        pytest.skip("NEPS is not installed")

    pipeline = Component(_A, name="hi", space={"a": (1, 10)})
    space = pipeline.search_space(parser=NEPSOptimizer.preferred_parser())
    return NEPSOptimizer.create(space=space, overwrite=True), "loss"


@parametrize_with_cases("optimizer, key_to_report_in", cases=".", prefix="opt_")
@parametrize("time_kind", [Timer.Kind.WALL, Timer.Kind.CPU, Timer.Kind.PROCESS])
@parametrize(
    "memory_unit",
    [
        Memory.Unit.BYTES,
        Memory.Unit.KILOBYTES,
        Memory.Unit.MEGABYTES,
        Memory.Unit.GIGABYTES,
    ],
)
def test_report_success(
    optimizer: Optimizer,
    time_kind: Timer.Kind,
    memory_unit: Memory.Unit,
    key_to_report_in: str,
) -> None:
    """Test that the optimizer can report a success."""
    trial = optimizer.ask()
    report = target_function(
        trial,
        time_kind=time_kind,
        mem_unit=memory_unit,
        err=None,
        key_to_report_in=key_to_report_in,
    )
    optimizer.tell(report)

    assert report.status == Trial.Status.SUCCESS
    assert valid_time_interval(report.time)
    assert report.trial.info is trial.info
    assert report.results == {key_to_report_in: 1}


@parametrize_with_cases("optimizer, key_to_report_in", cases=".", prefix="opt_")
@parametrize("time_kind", [Timer.Kind.WALL, Timer.Kind.CPU, Timer.Kind.PROCESS])
@parametrize(
    "memory_unit",
    [
        Memory.Unit.BYTES,
        Memory.Unit.KILOBYTES,
        Memory.Unit.MEGABYTES,
        Memory.Unit.GIGABYTES,
    ],
)
def test_report_failure(
    optimizer: Optimizer,
    time_kind: Timer.Kind,
    memory_unit: Memory.Unit,
    key_to_report_in: str,
):
    trial = optimizer.ask()
    report = target_function(
        trial,
        time_kind=time_kind,
        mem_unit=memory_unit,
        err=ValueError("Error inside Target Function"),
        key_to_report_in=key_to_report_in,
    )
    optimizer.tell(report)
    assert report.status is Trial.Status.FAIL

    assert valid_time_interval(report.time)
    assert isinstance(report.exception, ValueError)
    assert isinstance(report.traceback, str)
    assert report.results == {key_to_report_in: 2000}
