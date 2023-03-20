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

from byop.optimization.optimizer import Optimizer, Trial, TrialReport
from byop.optuna_space.space_parsing import OPTUNA_CONFIG, OPTUNA_SEARCH_SPACE

OPTUNA_OBJECTIVE = "minimize"


class OptunaOptimizer(Optimizer[OptunaTrial, OPTUNA_CONFIG]):
    """An optimizer that uses Optuna to optimize a search space."""

    def __init__(self, *, study: Study, space: OPTUNA_SEARCH_SPACE) -> None:
        """Initialize the optimizer.

        Args:
            study (Study): The Optuna Study to use.
            space (OPTUNA_SEARCH_SPACE): Defines the current search space.
        """
        self.study = study
        self.space = space

    @classmethod
    def create(
        cls,
        *,
        space: OPTUNA_SEARCH_SPACE,
        study_name: str | None = None,
        storage: str | BaseStorage | None = None,
        sampler: BaseSampler | None = None,
        pruner: BasePruner | None = None,
    ) -> Self:
        """Create a new Optuna optimizer. For more information, check
            Optuna documentation
            [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#).

        Args:
            space (OPTUNA_SEARCH_SPACE): Defines the current search space.
            study_name (str | None): Name of optuna study. If this argument is set to
                 None, a unique name is generated automatically
            storage (str | BaseStorage | None): Database URL. If this argument is set to
                 None, in-memory storage is used, and the Study will not be persistent.
            sampler (BaseSampler | None): A sampler object.
            pruner (BasePruner | None): A pruner object.

        Returns:
            Self: _description_
        """
        study = optuna.create_study(
            direction=OPTUNA_OBJECTIVE,
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

        return cls(study=study, space=space)

    def ask(self) -> Trial[OptunaTrial, OPTUNA_CONFIG]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        optuna_trial = self.study.ask(self.space)

        config = optuna_trial.params

        trial_number = optuna_trial.number
        unique_name = f"{trial_number=}"
        return Trial(name=unique_name, config=config, info=optuna_trial)

    def tell(self, report: TrialReport[OptunaTrial, OPTUNA_CONFIG]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        reported_costs = report.results.get("cost", None)

        trial = report.info
        trial_state = TrialState.COMPLETE if report.successful else TrialState.FAIL

        # In case of failure, Optuna does not expect any value
        if trial_state == TrialState.FAIL:
            reported_costs = None
        self.study.tell(trial=trial, values=reported_costs, state=trial_state)