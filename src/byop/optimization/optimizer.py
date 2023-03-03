"""Protocols for the optimization module."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Literal, Protocol

from byop.exceptions import attach_traceback
from byop.timing import TimeInterval, TimeKind, Timer
from byop.types import Config, TrialInfo


@dataclass
class TrialReport(Generic[TrialInfo, Config]):
    """A report of a trial.

    Attributes:
        info: The info of the trial.
        time: The time taken by the trial.
        exception: The exception raised by the trial, if any.
        successful: Whether the trial was successful.
        extra: Any extra information to attach to the report.
    """

    name: str
    config: Config
    info: TrialInfo
    time: TimeInterval
    exception: Exception | None
    successful: bool
    results: dict[str, Any] = field(default_factory=dict)


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

    def success(self, **results: Any) -> TrialReport[TrialInfo, Config]:
        """Mark the trial as successful.

        Will raise a `RuntimeError` if called before `begin`.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError("Cannot call `success` before `begin`")

        return TrialReport(
            name=self.name,
            config=self.config,
            info=self.info,
            time=self.time if self.time is not None else self.timer.stop(),
            exception=self.exception,
            successful=True,
            results=results,
        )

    def fail(self, **results: Any) -> TrialReport[TrialInfo, Config]:
        """Mark the trial as failed.

        Will raise a `RuntimeError` if called before `begin`.

        Returns:
            The result of the trial.
        """
        if self.timer is None:
            raise RuntimeError("Cannot call `success` before `begin`")

        return TrialReport(
            name=self.name,
            config=self.config,
            info=self.info,
            time=self.time if self.time is not None else self.timer.stop(),
            exception=self.exception,
            successful=False,
            results=results,
        )


class Optimizer(Protocol[TrialInfo, Config]):
    """An optimizer protocol.

    An optimizer is an object that can be asked for a trail using `ask` and a
    `tell` to inform the optimizer of the report from that trial.
    """

    def tell(self, report: TrialReport[TrialInfo, Config]) -> None:
        """Tell the optimizer the report for an asked trial.

        Args:
            report: The report of the trial.
        """
        ...

    def ask(self) -> Trial[TrialInfo, Config]:
        """Ask the optimizer for a trial to evaluate.

        Returns:
            A config to sample.
        """
        ...
