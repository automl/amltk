"""A trial for an optimization task.

TODO: Populate more here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
)
from typing_extensions import Concatenate, ParamSpec

from byop.events import Event, Subscriber
from byop.exceptions import attach_traceback
from byop.scheduling import (
    Scheduler,
    Task as TaskBase,
)
from byop.timing import TimeInterval, TimeKind, Timer

if TYPE_CHECKING:
    from concurrent.futures import Future

Info = TypeVar("Info")
"""The info associated with a trial"""

InfoInner = TypeVar("InfoInner")

P = ParamSpec("P")


class Trial(Generic[Info]):
    """A trial context manager."""

    def __init__(
        self,
        *,
        name: str,
        config: Mapping[str, Any],
        info: Info,
        time: TimeInterval | None = None,
        timer: Timer | None = None,
        exception: Exception | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the trial.

        Args:
            name: The name of the trial.
            config: The config for the trial.
            info: The info of the trial.
            time: The time taken by the trial.
            timer: The timer used to time the trial.
            exception: The exception raised by the trial, if any.
            seed: The seed to use if suggested by the optimizer.
        """
        self.name = name
        self.config = config
        self.info = info
        self.time = time
        self.timer = timer
        self.exception = exception
        self.seed = seed

    @contextmanager
    def begin(
        self,
        time: TimeKind | Literal["wall", "cpu", "process"] = "cpu",
    ) -> Iterator[None]:
        """Begin the trial.

        Will begin timing the trial in the `with` block, attaching the timings to the
        trial once completed.
        If an exception is raised, it will be attached to the trial as well,
        with the traceback attached to the actual error message, such that
        it can be pickled and sent back to the main process loop.

        Args:
            time: The timer kind to use for the trial.
        """
        self.timer = Timer.start(kind=time)
        try:
            yield
        except Exception as error:  # noqa: BLE001
            self.exception = attach_traceback(error)
        finally:
            if self.time is None:
                self.time = self.timer.stop()

    def success(self, **results: Any) -> Trial.SuccessReport[Info]:
        """Generate a success report.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError(
                "Cannot succeed a trial that has not been started."
                " Please use `with trial.begin():` to start the trial.",
            )

        time = self.time if self.time is not None else self.timer.stop()
        return Trial.SuccessReport(trial=self, time=time, results=results)

    def fail(self, **results: Any) -> Trial.FailReport[Info]:
        """Generate a failure report.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError(
                "Cannot fail a trial that has not been started."
                " Please use `with trial.begin():` to start the trial.",
            )

        time = self.time if self.time is not None else self.timer.stop()
        exception = self.exception
        return Trial.FailReport(
            trial=self,
            time=time,
            exception=exception,
            results=results,
        )

    def crashed(
        self,
        exception: BaseException | None = None,
    ) -> Trial.CrashReport[Info]:
        """Generate a crash report.

        Args:
            exception: The exception that caused the crash. If not provided, the
                exception will be taken from the trial. If this is still `None`,
                a `RuntimeError` will be raised.

        Returns:
            The result of the trial.
        """
        if exception is None and self.exception is None:
            raise RuntimeError(
                "Cannot generate a crash report without an exception."
                " Please provide an exception or use `with trial.begin():` to start"
                " the trial.",
            )

        exception = exception if exception else self.exception
        assert exception is not None

        return Trial.CrashReport(trial=self, exception=exception)

    @dataclass
    class Report(Generic[InfoInner]):
        """A report for a trial, one of Crash, Success or Fail.

        Attributes:
            trial: The trial that was run.
        """

        trial: Trial[InfoInner]

    @dataclass
    class CrashReport(Report[InfoInner]):
        """A report for a crashed trial.

        Attributes:
            trial: The trial that was run.
            exception: The exception for the trial.
        """

        trial: Trial[InfoInner]
        exception: BaseException

    @dataclass
    class SuccessReport(Report[InfoInner]):
        """A report for a successful trial.

        Attributes:
            trial: The trial that was run.
            time: The time taken by the trial.
            results: The results of the trial.
        """

        trial: Trial[InfoInner]
        time: TimeInterval
        results: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class FailReport(Report[InfoInner]):
        """A report for a failed trial.

        Attributes:
            trial: The trial that was run.
            time: The time taken by the trial.
            exception: The exception raised by the trial.
            results: The results of the trial.
        """

        trial: Trial[InfoInner]
        time: TimeInterval
        exception: BaseException | None
        results: dict[str, Any] = field(default_factory=dict)

    class Objective(Generic[P, InfoInner]):
        """Attach static information to a function to be optimized."""

        def __init__(
            self,
            f: Callable[Concatenate[Trial[InfoInner], P], Trial.Report[InfoInner]],
            *args: P.args,
            **kwargs: P.kwargs,
        ):
            """Initialize the objective.

            Args:
                f: The function to optimize.
                args: The positional arguments to pass to `f` after trial.
                kwargs: The keyword arguments to pass to `f`.
            """
            self.f = f
            self.args = args
            self.kwargs = kwargs

        def __call__(self, trial: Trial[InfoInner]) -> Trial.Report[InfoInner]:
            """Call the objective."""
            return self.f(trial, *self.args, **self.kwargs)

    class Task(TaskBase):
        """A task that will run a target function and tell the optimizer the result."""

        SUCCESS: Event[Trial.SuccessReport] = Event("trial-success")
        """The event that is triggered when a trial succeeds."""

        FAILURE: Event[Trial.FailReport] = Event("trial-failure")
        """The event that is triggered when a trial fails."""

        CRASHED: Event[Trial.CrashReport] = Event("trial-crashed")
        """The event that is triggered when a trial crashes."""

        REPORT: Event[Trial.Report] = Event("trial-report")
        """The event that is triggered when a trial reports anything."""

        def __init__(
            self,
            function: Callable[[Trial[InfoInner]], Trial.Report[InfoInner]],
            scheduler: Scheduler,
            *,
            name: str | None = None,
            call_limit: int | None = None,
            concurrent_limit: int | None = None,
            memory_limit: int | tuple[int, str] | None = None,
            cpu_time_limit: int | tuple[float, str] | None = None,
            wall_time_limit: int | tuple[float, str] | None = None,
        ) -> None:
            """Initialize a task.

            See [`Task`][byop.scheduling.task.Task] for more details.
            """
            super().__init__(
                function,
                scheduler,
                name=name,
                call_limit=call_limit,
                concurrent_limit=concurrent_limit,
                memory_limit=memory_limit,
                cpu_time_limit=cpu_time_limit,
                wall_time_limit=wall_time_limit,
            )
            self.trial_lookup: dict[Future, Trial] = {}

            self.on_f_returned(self._emit_report)
            self.on_f_exception(self._emit_report)

            self.on_report: Subscriber[Trial.Report[InfoInner]]
            self.on_report = self.subscriber(self.REPORT)

            self.on_failed: Subscriber[Trial.FailReport[InfoInner]]
            self.on_failed = self.subscriber(self.FAILURE)

            self.on_success: Subscriber[Trial.SuccessReport[InfoInner]]
            self.on_success = self.subscriber(self.SUCCESS)

            self.on_crashed: Subscriber[Trial.CrashReport[InfoInner]]
            self.on_crashed = self.subscriber(self.CRASHED)

        def __call__(
            self,
            trial: Trial[InfoInner],
        ) -> Future[Trial.Report[InfoInner]] | None:
            """Run the trial and return the future for the result.

            Args:
                trial: The trial to run.

            Returns:
                The future for the result of the trial.
            """
            future = super().__call__(trial)
            if future is not None:
                self.trial_lookup[future] = trial

            return future

        def _emit_report(
            self,
            future: Future,
            report: Trial.Report | BaseException,
        ) -> None:
            """Emit a report for a trial based on the type of the report."""
            # Emit the fact a report happened
            if isinstance(report, BaseException):
                report = self.trial_lookup[future].crashed(report)

            emit_items: dict[Event, Any] = {
                self.REPORT: ((report,), None),
            }

            # Emit the specific type of report
            event: Event
            if isinstance(report, Trial.SuccessReport):
                event = self.SUCCESS
            elif isinstance(report, Trial.FailReport):
                event = self.FAILURE
            elif isinstance(report, Trial.CrashReport):
                event = self.CRASHED
            else:
                raise TypeError(f"Unexpected report type: {type(report)}")

            emit_items[event] = ((report,), None)
            self.emit_many(emit_items)  # type: ignore
