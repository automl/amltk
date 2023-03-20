"""Ask-and-tell controller.

This controller will run a target function in a scheduler and
then tell the optimizer the result of the trial after each trial.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from byop.optimization import Optimizer, Trial
from byop.scheduling import Scheduler
from byop.types import Config

logger = logging.getLogger(__name__)

P = ParamSpec("P")
Q = ParamSpec("Q")
FailT = TypeVar("FailT")
SuccessT = TypeVar("SuccessT")
Info = TypeVar("Info")


class AskAndTell(Generic[Info, Config]):
    """A controller that will run a target function and tell the optimizer
    the result.
    """

    def __init__(
        self,
        *,
        objective: Callable[[Trial[Info]], Trial.Report[Info]],
        optimizer: Optimizer[Info],
        scheduler: Scheduler | None = None,
        max_trials: int | None = None,
        concurrent_trials: int = 1,
    ):
        """Initialize the controller.

        Args:
            objective: The objective to run.
            optimizer: The optimizer to use.
            scheduler: The scheduler to use.
                When None, a basic Scheduler is set up that runs with a single process
                worker.
            max_trials: The maximum number of trials to run.
                If not provided, will run indefinitely.
            concurrent_trials: The number of concurrent trials to run.
                Defaults to 1.
        """
        if concurrent_trials < 1:
            raise ValueError(f"{concurrent_trials=} must be >= 1")

        if scheduler is None:
            scheduler = Scheduler.with_processes(max_workers=1)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.events = scheduler.event_manager
        self.max_trials = max_trials
        self.concurrent_trials = concurrent_trials
        self._objective = objective

        self.trial = Trial.Task(
            self._objective,
            scheduler,
            name="trial",
            call_limit=max_trials,
        )

        # Set up the scheduler to ask and evaluate new trials when
        # the scheduler starts
        scheduler.on_start(self.ask_and_evaluate, repeat=concurrent_trials)

        # Whenever a trial generates a report, ask and begin a new trial
        self.trial.on_report(self.ask_and_evaluate, name="ask-and-evaluate")

        # Whenever we get something from a trial, tell the optimizer about it.
        self.trial.on_report(self.optimizer.tell, name="tell-optimizer")

        # Dictionary from a trial name to the trial itself
        self.trial_lookup: dict[str, Trial[Info]] = {}

    def ask_and_evaluate(self, *_: Any) -> None:
        """Ask the optimizer for a new trial and evaluate it."""
        trial = self.optimizer.ask()
        logger.info(f"Running {trial.name}")
        logger.debug(f"{trial=}")
        self.trial(trial)

    def run(
        self,
        *,
        timeout: float | None = None,
        wait: bool = True,
        end_on_empty: bool = True,
    ) -> Scheduler.ExitCode:
        """Run the controller.

        Args:
            timeout: The maximum time to run for.
            wait: Whether to wait for the scheduler to finish.
            end_on_empty: Whether to end the scheduler when the queue is empty.
        """
        return self.scheduler.run(end_on_empty=end_on_empty, timeout=timeout, wait=wait)
