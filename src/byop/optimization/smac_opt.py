"""A thin wrapper around SMAC to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from ConfigSpace import Configuration, ConfigurationSpace
from pynisher import MemoryLimitException, TimeoutException
from smac import HyperparameterOptimizationFacade, Scenario
from smac.facade import AbstractFacade
from smac.runhistory import StatusType
from smac.runhistory import TrialInfo as SMACTrialInfo
from smac.runhistory import TrialValue as SMACTrialValue
from typing_extensions import Self

from byop.optimization.optimizer import (
    CrashReport,
    FailReport,
    Optimizer,
    SuccessReport,
    Trial,
    TrialReport,
)
from byop.randomness import as_int
from byop.types import Seed


class SMACOptimizer(Optimizer[SMACTrialInfo, Configuration]):
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
        print(seed)
        facade = HyperparameterOptimizationFacade(
            scenario=Scenario(configspace=space, seed=seed),
            target_function="dummy",  # NOTE: https://github.com/automl/SMAC3/issues/946
            overwrite=not continue_from_last_run,
            logging_level=logging_level,
        )
        return cls(facade=facade)

    def ask(self) -> Trial[SMACTrialInfo, Configuration]:
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
        unique_name = f"{config_id=}.{instance=}.{seed=}.{budget=}"
        return Trial(name=unique_name, config=config, info=smac_trial_info)

    def tell(  # noqa: C901
        self,
        report: TrialReport[SMACTrialInfo, Configuration],
    ) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        # If we're successful, get the cost and times and report them
        if isinstance(report, SuccessReport):
            if "cost" not in report.results:
                raise ValueError(
                    f"Report must have 'cost' if successful but got {report}."
                    " Use `trial.success(cost=...)` to set the results of the trial."
                )

            trial_value = SMACTrialValue(
                time=report.time.duration,
                starttime=report.time.start,
                endtime=report.time.end,
                cost=report.results["cost"],
                status=StatusType.SUCCESS,
                additional_info=report.results.get("additional_info", {}),
            )
            return

        if isinstance(report, FailReport):
            duration = report.time.duration
            start = report.time.start
            end = report.time.end
            reported_cost = report.results.get("cost", None)
            additional_info = report.results.get("additional_info", {})
        elif isinstance(report, CrashReport):
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
        if report.exception is not None:
            status_type = status_types.get(type(report.exception), StatusType.CRASHED)
            additional_info["exception"] = report.exception

        # If we have no reported costs, we need to ensure that we have a
        # valid crash_cost based on the number of objectives
        crash_cost = self.facade.scenario.crash_cost
        objectives = self.facade.scenario.objectives

        if reported_cost is not None:
            cost = reported_cost
        elif isinstance(crash_cost, float) and not isinstance(objectives, Sequence):
            cost = crash_cost
        elif isinstance(crash_cost, float) and isinstance(objectives, Sequence):
            cost = [crash_cost for _ in range(len(objectives))]
        elif isinstance(crash_cost, Sequence) and isinstance(objectives, Sequence):
            cost = crash_cost
        else:
            raise ValueError(
                f"Multiple crash cost reported ({crash_cost}) for only a single"
                f" objective in `Scenario({objectives=}, ...)"
            )

        if (isinstance(cost, Sequence) and isinstance(objectives, Sequence)) and (
            len(cost) != len(objectives)
        ):
            raise ValueError(
                f"Length of crash cost ({len(cost)}) and objectives "
                f"({len(objectives)}) must be equal"
            )

        trial_value = SMACTrialValue(
            time=duration,
            starttime=start,
            endtime=end,
            cost=cost,
            status=status_type,
            additional_info=additional_info,
        )
        self.facade.tell(info=report.info, value=trial_value, save=True)
