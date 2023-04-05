"""A thin wrapper around Optuna to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

from typing import Any, Sequence
from typing_extensions import Self

import optuna
from optuna.study import Study, StudyDirection
from optuna.trial import (
    Trial as OptunaTrial,
    TrialState,
)

from byop.optimization import Optimizer, Trial
from byop.optuna.space import OptunaSearchSpace


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
        **kwargs: Any,
    ) -> Self:
        """Create a new Optuna optimizer. For more information, check Optuna
            documentation
            [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#).

        Args:
            space: Defines the current search space.
            **kwargs: Additional arguments to pass to
                [`optuna.create_study`][optuna.create_study].

        Returns:
            Self: The newly created optimizer.
        """
        study = optuna.create_study(**kwargs)
        return cls(study=study, space=space)

    def ask(self) -> Trial[OptunaTrial]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        optuna_trial = self.study.ask(self.space)
        config = optuna_trial.params
        trial_number = optuna_trial.number
        unique_name = f"{trial_number=}"
        return Trial(name=unique_name, config=config, info=optuna_trial)

    def tell(self, report: Trial.Report[OptunaTrial]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        trial = report.trial.info

        if isinstance(report, Trial.SuccessReport):
            trial_state = TrialState.COMPLETE
            values = self._verify_success_report_values(report)
        else:
            trial_state = TrialState.FAIL
            values = None

        self.study.tell(trial=trial, values=values, state=trial_state)

    def _verify_success_report_values(
        self,
        report: Trial.SuccessReport[OptunaTrial],
    ) -> float | Sequence[float]:
        """Verify that the report is valid.

        Args:
            report: The report to check.

        Raises:
            ValueError: If both "cost" and "values" reported or
                if the study direction is not "minimize" and "cost" is reported.
        """
        if "cost" in report.results and "values" in report.results:
            raise ValueError(
                "Both 'cost' and 'values' were provided in the report. "
                "Only one of them should be provided."
            )

        if "cost" not in report.results and "values" not in report.results:
            raise ValueError(
                "Neither 'cost' nor 'values' were provided in the report. "
                "At least one of them should be provided."
            )

        directions = self.study.directions

        values = None
        if "cost" in report.results:
            if not all(direct == StudyDirection.MINIMIZE for direct in directions):
                raise ValueError(
                    "The study direction is not 'minimize',"
                    " but 'cost' was provided in the report."
                )
            values = report.results["cost"]
        else:
            values = report.results["values"]

        if not (
            isinstance(values, (float, int))
            or (
                isinstance(values, Sequence)
                and all(isinstance(value, (float, int)) for value in values)
            )
        ):
            raise ValueError(
                f"Reported {values=} should be float or a sequence of floats"
            )

        return values
