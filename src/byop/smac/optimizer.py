"""A thin wrapper around SMAC to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Sequence

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import (
    StatusType,
    TrialInfo as SMACTrialInfo,
    TrialValue as SMACTrialValue,
)

from byop.optimization import Optimizer, Trial
from byop.randomness import as_int
from pynisher import MemoryLimitException, TimeoutException

if TYPE_CHECKING:
    from pathlib import Path
    from typing_extensions import Self

    from ConfigSpace import ConfigurationSpace
    from smac.facade import AbstractFacade

    from byop.types import Seed


logger = logging.getLogger(__name__)


class SMACOptimizer(Optimizer[SMACTrialInfo]):
    """An optimizer that uses SMAC to optimize a config space."""

    def __init__(self, *, facade: AbstractFacade) -> None:
        """Initialize the optimizer.

        Args:
            facade: The SMAC facade to use.
        """
        self.facade = facade

    @classmethod
    def HPO(  # noqa: N802
        cls,
        *,
        space: ConfigurationSpace,
        seed: Seed | None = None,
        continue_from_last_run: bool = False,
        logging_level: int | Path | Literal[False] | None = False,
    ) -> Self:
        """Create a new SMAC optimizer using the HPO facade.

        Args:
            space: The config space to optimize.
            seed: The seed to use for the optimizer.
            continue_from_last_run: Whether to continue from a previous run.
            logging_level: The logging level to use.
                This argument is passed forward to SMAC, use False to disable
                SMAC's handling of logging.
        """
        seed = as_int(seed)
        facade = HyperparameterOptimizationFacade(
            scenario=Scenario(configspace=space, seed=seed),
            target_function="dummy",  # NOTE: https://github.com/automl/SMAC3/issues/946
            overwrite=not continue_from_last_run,
            logging_level=logging_level,
        )
        return cls(facade=facade)

    def ask(self) -> Trial[SMACTrialInfo]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        smac_trial_info = self.facade.ask()
        config = smac_trial_info.config
        budget = smac_trial_info.budget
        instance = smac_trial_info.instance
        seed = smac_trial_info.seed

        config_id = self.facade.runhistory.config_ids[config]
        unique_name = f"{config_id=}_{seed=}_{budget=}_{instance=}"
        trial: Trial[SMACTrialInfo] = Trial(
            name=unique_name,
            config=dict(config),
            info=smac_trial_info,
            seed=seed,
        )
        logger.info(f"Asked for trial {trial.name}")
        logger.debug(f"{trial=}")
        return trial

    def tell(self, report: Trial.Report[SMACTrialInfo]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        logger.info(f"Telling report for trial {report.trial.name}")
        logger.debug(f"{report=}")
        # If we're successful, get the cost and times and report them
        if isinstance(report, Trial.SuccessReport):
            if "cost" not in report.results:
                raise ValueError(
                    f"Report must have 'cost' if successful but got {report}."
                    " Use `trial.success(cost=...)` to set the results of the trial.",
                )

            trial_value = SMACTrialValue(
                time=report.time.duration,
                starttime=report.time.start,
                endtime=report.time.end,
                cost=report.results["cost"],
                status=StatusType.SUCCESS,
                additional_info=report.results.get("additional_info", {}),
            )
            self.facade.tell(info=report.trial.info, value=trial_value, save=True)
            return

        if isinstance(report, Trial.FailReport):
            duration = report.time.duration
            start = report.time.start
            end = report.time.end
            reported_cost = report.results.get("cost", None)
            additional_info = report.results.get("additional_info", {})
        else:
            duration = 0
            start = 0
            end = 0
            reported_cost = None
            additional_info = {}

        # We got either a fail or a crash, time to deal with it
        status_types: dict[type, StatusType] = {
            MemoryLimitException: StatusType.MEMORYOUT,
            TimeoutException: StatusType.TIMEOUT,
        }
        status_type = StatusType.CRASHED

        assert isinstance(report, (Trial.FailReport, Trial.CrashReport))
        if report.exception is not None:
            status_type = status_types.get(type(report.exception), StatusType.CRASHED)
            additional_info["exception"] = str(report.exception)
            additional_info["traceback"] = report.traceback

        # If we have no reported costs, we need to ensure that we have a
        # valid crash_cost based on the number of objectives
        crash_cost = self.facade.scenario.crash_cost
        objectives = self.facade.scenario.objectives

        cost: float | list[float]

        if reported_cost is not None:
            cost = reported_cost
        elif isinstance(crash_cost, float):
            cost = crash_cost
        elif isinstance(crash_cost, float):
            cost = [crash_cost for _ in range(len(objectives))]
        elif isinstance(crash_cost, Sequence):
            cost = list(crash_cost)
        else:
            raise ValueError(
                f"Multiple crash cost reported ({crash_cost}) for only a single"
                f" objective in `Scenario({objectives=}, ...)",
            )

        if isinstance(cost, Sequence) and (len(cost) != len(objectives)):
            raise ValueError(
                f"Length of crash cost ({len(cost)}) and objectives "
                f"({len(objectives)}) must be equal",
            )

        trial_value = SMACTrialValue(
            time=duration,
            starttime=start,
            endtime=end,
            cost=cost,
            status=status_type,
            additional_info=additional_info,
        )
        self.facade.tell(info=report.trial.info, value=trial_value, save=True)
