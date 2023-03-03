"""A thin wrapper around SMAC to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from smac.facade import AbstractFacade
from smac.runhistory import StatusType
from smac.runhistory import TrialInfo as SMACTrialInfo
from smac.runhistory import TrialValue as SMACTrialValue
from typing_extensions import Self

from byop.optimization.optimizer import Optimizer, Trial, TrialReport
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

    def tell(self, report: TrialReport[SMACTrialInfo, Configuration]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        # TODO: We do not support memory and time limiting yet
        reported_costs = report.results.get("cost", None)
        if reported_costs is None:
            raise ValueError("No cost (float | list[float]) reported for trial")

        if report.successful:
            status_type = StatusType.SUCCESS
            trial_value = SMACTrialValue(
                time=report.time.duration,
                starttime=report.time.start,
                endtime=report.time.end,
                cost=reported_costs,
                status=status_type,
                additional_info=report.results.get("additional_info", {}),
            )
        else:
            status_type = StatusType.CRASHED
            crash_cost = self.facade.scenario.crash_cost

            # Check to make sure we can convert crash_cost to that of the reported costs
            if (
                isinstance(reported_costs, Sequence)
                and isinstance(crash_cost, Sequence)
            ) and (len(reported_costs) != len(crash_cost)):
                msg = (
                    f"Length of reported costs ({len(reported_costs)}) and crash costs "
                    f"({len(crash_cost)}) must be equal"
                )
                raise ValueError(msg)

            if isinstance(reported_costs, Sequence) and not isinstance(
                crash_cost, Sequence
            ):
                crash_cost = [crash_cost] * len(reported_costs)

            trial_value = SMACTrialValue(
                time=report.time.duration,
                starttime=report.time.start,
                endtime=report.time.end,
                cost=crash_cost,
                status=status_type,
                additional_info=report.results.get("additional_info", {}),
            )

        self.facade.tell(info=report.info, value=trial_value, save=True)
