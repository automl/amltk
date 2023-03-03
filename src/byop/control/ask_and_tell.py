"""Ask-and-tell controller.

This controller will run a target function in a scheduler and
then tell the optimizer the result of the trial after each trial.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum, auto
import logging
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from byop.optimization import Optimizer, Trial, TrialReport
from byop.scheduling import ExitCode, Scheduler, Task, TaskFuture
from byop.types import CallbackName, Config, TrialInfo

logger = logging.getLogger(__name__)

P = ParamSpec("P")
Q = ParamSpec("Q")
FailT = TypeVar("FailT")
SuccessT = TypeVar("SuccessT")


class AskAndTellEvent(Enum):
    """The specific events that can be emitted by the ask-and-tell controller."""

    SUCCESS = auto()
    """Emitted when a trial succeeds."""

    FAILURE = auto()
    """Emmited when a trial fails."""


class AskAndTellTask(Task[[Trial[TrialInfo, Config]], TrialReport[TrialInfo, Config]]):
    """A task that will run a target function and tell the optimizer the result."""

    def on_success(
        self,
        callback: Callable[[TrialReport[TrialInfo, Config]], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter], bool] | None = None,
    ) -> None:
        """Register a callback to be called when a trial succeeds.

        Args:
            callback: The callback to call.
            name: The name of the callback.
            when: A function that takes the counter of events and returns
                whether the callback should be called.
        """
        self.scheduler.on(AskAndTellEvent.SUCCESS, callback, name=name, when=when)

    def on_failure(
        self,
        callback: Callable[[TrialReport[TrialInfo, Config]], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter], bool] | None = None,
    ) -> None:
        """Register a callback to be called when a trial fails.

        Args:
            callback: The callback to call.
            name: The name of the callback.
            when: A function that takes the counter of events and returns
                whether the callback should be called.
        """
        self.scheduler.on(AskAndTellEvent.FAILURE, callback, name=name, when=when)


class AskAndTell(Generic[TrialInfo, Config]):
    """A controller that will run a target function and tell the optimizer
    the result.
    """

    def __init__(
        self,
        *,
        objective: Callable[[Trial[TrialInfo, Config]], TrialReport[TrialInfo, Config]],
        optimizer: Optimizer[TrialInfo, Config],
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
        self.events = scheduler.event_manager
        self.max_trials = max_trials
        self.concurrent_trials = concurrent_trials
        self._objective = objective

        self.trial = scheduler.task(
            self._objective,
            name="trial",
            limit=max_trials,
            task_type=AskAndTellTask,
        )

        # Whenever a worker is done with a task, check if we have
        # a return value and if we do, raise further events.
        self.trial.on_done(self._maybe_emit_success_or_fail)

        # Whenever a trial is done, either returned or not, ask for a new trial
        # and evaluate it.
        self.trial.on_done(self._ask_and_evaluate)

        # Whenever we get a return value from the objective, tell the
        # optimizer.
        self.trial.on_return(self.optimizer.tell)

        for i in range(concurrent_trials):
            scheduler.on_start(self._ask_and_evaluate, name=f"worker-{i}")

    def _maybe_emit_success_or_fail(
        self,
        task_future: TaskFuture[..., TrialReport[TrialInfo, Config]],
    ) -> None:
        # We only care about emitting a success or fail of a trial
        # if the call to the objective actually returned a value.
        if task_future.has_result():
            task_name = task_future.name
            report: TrialReport[TrialInfo, Config] = task_future.result
            success = report.successful

            event = AskAndTellEvent.SUCCESS if success else AskAndTellEvent.FAILURE
            self.events.emit(event, report)
            self.events.emit((task_name, event), report)

    def _ask_and_evaluate(self, *_: Any) -> None:
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
    ) -> ExitCode:
        """Run the controller.

        Args:
            timeout: The maximum time to run for.
            wait: Whether to wait for the scheduler to finish.
            end_on_empty: Whether to end the scheduler when the queue is empty.
        """
        return self.scheduler.run(end_on_empty=end_on_empty, timeout=timeout, wait=wait)
