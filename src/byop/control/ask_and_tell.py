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
from byop.types import CallbackName, Config, TaskName, TrialInfo

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
            call_limit=max_trials,
            task_type=AskAndTellTask,
        )

        # Whenever a trial is done, either returned or not, ask for a new trial
        # and evaluate it.
        self.trial.on_done(self.when_done_ask_and_evaluate, name="ask-and-evaluate")

        # Whenever we get something from a trial, tell the optimizer about it.
        # This will also emit appropriate events.
        self.trial.on_done(self.when_done_tell_optimizer, name="tell-optimizer")

        # Set up the scheduler to run the callback several times
        for i in range(concurrent_trials):
            scheduler.on_start(
                self.when_done_ask_and_evaluate,
                name=f"ask-and-tell-worker-{i}",
            )

        self.trial_lookup: dict[TaskName, Trial[TrialInfo, Config]] = {}

    def when_done_tell_optimizer(
        self,
        task_future: TaskFuture[
            [Trial[TrialInfo, Config]], TrialReport[TrialInfo, Config]
        ],
    ) -> None:
        """Tell the optimizer about the results of a trial.

        Args:
            task_future: The future for the task that completed.
        """
        if task_future not in self.trial_lookup:
            raise RuntimeError(
                f"Task {task_future} recorded as done but the controller has no"
                " record of it. Please raise an issue on github!"
            )

        if task_future.cancelled():
            # There's really nothing we can do of value if it was cancelled.
            # Telling the optimizer about it is misleading as it's not due
            # to the trial itself. There's also no SUCCESS or FAILURE state
            # to really report to the user. They can use the `on_cancelled`
            # callback to handle this.
            return

        if task_future.has_result():
            report = task_future.result
        else:
            trial = self.trial_lookup[task_future.name]
            if task_future.exception is not None:
                report = trial.crashed(exception=task_future.exception)
            else:
                raise RuntimeError(
                    f"Task {task_future} has no result or exception we can use."
                    " Otherwise the task should have had a result or been cancelled."
                    " Please raise an issue on github!"
                )

        self.optimizer.tell(report)

        task_name = task_future.name
        if report.successful:
            event = AskAndTellEvent.SUCCESS
        else:
            event = AskAndTellEvent.FAILURE

        self.events.emit(event, report)
        self.events.emit((task_name, event), report)

        if not task_future.cancelled():
            raise RuntimeError(
                f"Task {task_future} has neither a result nor an exception."
                " Please raise an issue on github!"
            )

    def when_done_ask_and_evaluate(self, *_: Any) -> None:
        """Ask the optimizer for a new trial and evaluate it."""
        trial = self.optimizer.ask()
        logger.info(f"Running {trial.name}")
        logger.debug(f"{trial=}")
        future = self.trial(trial)

        # If there is a future for the task, make sure we can look up the trial
        # that spawned it for later
        if future is not None:
            self.trial_lookup[future.name] = trial

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
