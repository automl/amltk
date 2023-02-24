"""Ask-and-tell controller.

This controller will run a target function in a scheduler and
then tell the optimizer the result of the trial after each trial.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Concatenate, Generic, ParamSpec

from byop.optimization import Optimizer
from byop.scheduling import ExitCode, Scheduler
from byop.types import Config, TrialResult

logger = logging.getLogger(__name__)

P = ParamSpec("P")


class Objective(Generic[Config, TrialResult]):
    """A callable that holds a target function and adds in the specific
    trial number and config when called.
    """

    def __init__(
        self,
        target_function: Callable[Concatenate[int, Config, P], TrialResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Initialize the objective.

        Args:
            target_function: The target function to run.
            args: The positional arguments to pass to the target function.
            kwargs: The keyword arguments to pass to the target function.
        """
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self,
        trial_number: int,
        config: Config,
    ) -> TrialResult:
        """Call the target function with the trial number and config.

        Args:
            trial_number: The trial number.
            config: The config to run the trial with.

        Returns:
            The result of the target function.
        """
        return self.target_function(trial_number, config, *self.args, **self.kwargs)


class AskAndTell(Generic[Config, TrialResult]):
    """A controller that will run a target function and tell the optimizer
    the result.
    """

    def __init__(
        self,
        *,
        objective: Objective[Config, TrialResult],
        optimizer: Optimizer[Config, TrialResult],
        scheduler: Scheduler,
        max_trials: int | None = None,
        concurrent_trials: int = 1,
    ):
        """Initialize the controller.

        Args:
            objective: The objective to run.
            optimizer: The optimizer to use.
            scheduler: The scheduler to use.
            max_trials: The maximum number of trials to run.
                If not provided, will run indefinitely.
            concurrent_trials: The number of concurrent trials to run.
                Defaults to 1.
        """
        if concurrent_trials < 1:
            raise ValueError(f"{concurrent_trials=} must be >= 1")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_trials = max_trials
        self.concurrent_trials = concurrent_trials
        self._objective = objective

        self.trial = scheduler.task(self._objective, name="trial", limit=max_trials)
        self.trial.on_done(self._ask_and_evaluate)
        self.trial.on_success(self.optimizer.tell)

        for i in range(concurrent_trials):
            scheduler.on_start(self._ask_and_evaluate, name=f"worker-{i}")

    def _ask_and_evaluate(self, *_: Any) -> None:
        config = self.optimizer.ask()
        trial_number = self.trial.n_called
        logger.debug(f"Running {trial_number=} with {config=}")
        self.trial(trial_number, config)

    @classmethod
    def objective(
        cls,
        target_function: Callable[Concatenate[int, Config, P], TrialResult],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Objective[Config, TrialResult]:
        """Create an objective from a target function, specific to AskAndTell.

        Args:
            target_function: The target function to run.
            args: The positional arguments to pass to the target function.
            kwargs: The keyword arguments to pass to the target function.

        Returns:
            An objective that can be passed to the AskAndTell controller.
        """
        return Objective(target_function, *args, **kwargs)

    def run(
        self,
        *,
        timeout: float | None = None,
        wait: bool = True,
        end_on_empty: bool = True,
    ) -> ExitCode:
        """Run the controller.

        Args:
            timeout: The maximum time to run for.
            wait: Whether to wait for the scheduler to finish.
            end_on_empty: Whether to end the scheduler when the queue is empty.
        """
        return self.scheduler.run(end_on_empty=end_on_empty, timeout=timeout, wait=wait)
