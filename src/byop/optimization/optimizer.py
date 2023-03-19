"""Protocols for the optimization module."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Iterator, Literal, Protocol, Union

from typing_extensions import TypeAlias

from byop.exceptions import attach_traceback
from byop.timing import TimeInterval, TimeKind, Timer
from byop.types import Config, TrialInfo


@dataclass
class SuccessReport(Generic[TrialInfo, Config]):
    """A report for a successful trial.

    Attributes:
        name: The name of the trial.
        config: The config of the trial.
        info: The info of the trial.
        time: The time taken by the trial.
        results: The results of the trial.
    """

    successful: ClassVar[Literal[True]] = True

    name: str
    config: Config
    info: TrialInfo
    time: TimeInterval
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailReport(Generic[TrialInfo, Config]):
    """A report for a failed trial.

    Attributes:
        name: The name of the trial.
        config: The config of the trial.
        info: The info of the trial.
        time: The time taken by the trial.
        exception: The exception raised by the trial.
        results: The results of the trial.
    """

    successful: ClassVar[Literal[False]] = False

    name: str
    config: Config
    info: TrialInfo
    time: TimeInterval
    exception: BaseException | None
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrashReport(Generic[TrialInfo, Config]):
    """A report for a crashed trial.

    Attributes:
        name: The name of the trial.
        info: The info of the trial.
        config: The config of the trial.
        exception: The exception for the trial.
    """

    successful: ClassVar[Literal[False]] = False

    name: str
    info: TrialInfo
    config: Config
    exception: BaseException


TrialReport: TypeAlias = Union[
    SuccessReport[TrialInfo, Config],
    FailReport[TrialInfo, Config],
    CrashReport[TrialInfo, Config],
]


@dataclass
class Trial(Generic[TrialInfo, Config]):
    """A trial context manager.

    Attributes:
        name: The name of the trial.
        info: The info of the trial.
        time: The time taken by the trial.
        exception: The exception raised by the trial, if any.
    """

    name: str
    config: Config
    info: TrialInfo
    time: TimeInterval | None = None
    timer: Timer | None = None
    exception: Exception | None = None

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

    def success(self, **results: Any) -> SuccessReport[TrialInfo, Config]:
        """Generate a success report.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError(
                "Cannot succeed a trial that has not been started."
                " Please use `with trial.begin():` to start the trial."
            )

        return SuccessReport(
            name=self.name,
            config=self.config,
            info=self.info,
            time=self.time if self.time is not None else self.timer.stop(),
            results=results,
        )

    def fail(self, **results: Any) -> FailReport[TrialInfo, Config]:
        """Generate a failure report.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError(
                "Cannot fail a trial that has not been started."
                " Please use `with trial.begin():` to start the trial."
            )

        return FailReport(
            name=self.name,
            config=self.config,
            info=self.info,
            time=self.time if self.time is not None else self.timer.stop(),
            exception=self.exception,
            results=results,
        )

    def crashed(
        self,
        exception: BaseException | None = None,
    ) -> CrashReport[TrialInfo, Config]:
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
                " the trial."
            )

        exception = exception if exception else self.exception
        assert exception is not None

        return CrashReport(
            name=self.name,
            config=self.config,
            info=self.info,
            exception=exception,
        )


class Optimizer(Protocol[TrialInfo, Config]):
    """An optimizer protocol.

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """

    def tell(self, report: TrialReport[TrialInfo, Config]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report for a trial
        """
        ...

    def ask(self) -> Trial[TrialInfo, Config]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
