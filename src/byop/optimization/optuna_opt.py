"""A thin wrapper around Optuna to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import Trial as OptunaTrial
from optuna.trial import TrialState
from typing_extensions import Self

from byop.optimization.optimizer import Optimizer, Trial
from byop.optuna_space.space_parsing import OptunaSearchSpace


class OptunaOptimizer(Optimizer[OptunaTrial]):
    """An optimizer that uses Optuna to optimize a search space."""

    def __init__(self, *, study: Study, space: OptunaSearchSpace) -> None:
        """Initialize the optimizer.

        Args:
            study: The Optuna Study to use.
            space: Defines the current search space.
        """
        self.study = study
        self.space = space

    @classmethod
    def create(
        cls,
        *,
        space: OptunaSearchSpace,
        study_name: str | None = None,
        storage: str | BaseStorage | None = None,
        sampler: BaseSampler | None = None,
        pruner: BasePruner | None = None,
        direction: str = "minimize",
    ) -> Self:
        """Create a new Optuna optimizer. For more information, check
            Optuna documentation
            [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#).

        Args:
            space: Defines the current search space.
            study_name: Name of optuna study. If this argument is set to
                 None, a unique name is generated automatically
            storage: Database URL. If this argument is set to
                 None, in-memory storage is used, and the Study will not be persistent.
            sampler: A sampler object.
            pruner: A pruner object.
            direction: Direction of optimization. Either 'minimize' or 'maximize'.

        Returns:
            Self: The newly created optimizer.
        """
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        return cls(study=study, space=space)

    def ask(self) -> Trial[OptunaTrial]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        optuna_trial = self.study.ask(self.space)
        trial_number = optuna_trial.number
        unique_name = f"{trial_number=}"
        return Trial(name=unique_name, info=optuna_trial)

    def tell(self, report: Trial.Report[OptunaTrial]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        if isinstance(report, Trial.SuccessReport):
            reported_costs = report.results.get("cost", None)
            trial = report.trial.info
            trial_state = TrialState.COMPLETE
        else:
            reported_costs = None
            trial = report.trial.info
            trial_state = TrialState.FAIL

        self.study.tell(trial=trial, values=reported_costs, state=trial_state)
