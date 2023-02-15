"""This module holds the definition of a Task.

A Task is a unit of work that can be scheduled by the scheduler. It is
defined by its name, its function, and it's `Future` representing the
final outcome of the task.

There is also the `CommTask` which follows the same interface but
additionally holds a `Comm` object that is used to communicate back
and forth with the remote worker.

TODO: More docs, here and on the classes
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import Callable, Generator, Generic, TypeVar
from uuid import uuid4

from typing_extensions import Self

from byop.event_manager import EventManager
from byop.scheduling.comm import Comm
from byop.scheduling.events import TaskStatus
from byop.types import CallbackName, Msg, TaskName

R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class Task(Generic[R]):
    """A task is a unit of work that can be scheduled by the scheduler."""

    name: TaskName
    """The name of the task"""

    future: asyncio.Future[R]
    """The future result of the task"""

    events: EventManager = field(repr=False)
    """The eventmanager it will emit to"""

    def on_submit(
        self,
        f: Callable[[Self], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self], bool] | None = None,
    ) -> None:
        """Called when the task is submitted to the scheduler."""
        name = name if name else f"{self.name}-submitted-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.SUBMITTED), f, name=name, pred=when)

    def on_finish(
        self,
        f: Callable[[Self], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self], bool] | None = None,
    ) -> None:
        """Called when the task is finished."""
        name = name if name else f"{self.name}-finish-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.FINISHED), f, name=name, pred=when)

    def on_success(
        self,
        f: Callable[[R], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[R], bool] | None = None,
    ) -> None:
        """Called when the task is successfully completed."""
        name = name if name else f"{self.name}-success-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.SUCCESS), f, name=name, pred=when)

    def on_error(
        self,
        f: Callable[[BaseException], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[BaseException], bool] | None = None,
    ) -> None:
        """Called when the task is finished."""
        name = name if name else f"{self.name}-error-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.ERROR), f, name=name, pred=when)

    def on_cancelled(
        self,
        f: Callable[[Self], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self], bool] | None = None,
    ) -> None:
        """Called when the task is finished."""
        name = name if name else f"{self.name}-cancelled-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.CANCELLED), f, name=name, pred=when)

    def cancel(self) -> None:
        """Cancel the task."""
        self.future.cancel()

    def done(self) -> bool:
        """Return True if the task is done."""
        return self.future.done()

    def result(self) -> R:
        """Return the result of the task."""
        return self.future.result()

    def exception(self) -> BaseException | None:
        """Return the exception of the task."""
        return self.future.exception()

    def _finish_up(self) -> None:
        if self.future.cancelled():
            logger.debug(f"Task {self} was cancelled")

            self.events.emit(TaskStatus.CANCELLED, self)
            self.events.emit((self.name, TaskStatus.CANCELLED), self)

        else:
            exception = self.future.exception()
            result = self.future.result() if exception is None else None

            logger.debug(f"Task {self} finished")
            self.events.emit(TaskStatus.FINISHED, self)
            self.events.emit((self.name, TaskStatus.FINISHED), self)

            if exception is None:
                logger.debug(f"Task {self} completed successfully")
                self.events.emit(TaskStatus.SUCCESS, result)
                self.events.emit((self.name, TaskStatus.SUCCESS), result)
            else:
                logger.debug(f"Task {self} failed with {exception}")
                self.events.emit(TaskStatus.ERROR, exception)
                self.events.emit((self.name, TaskStatus.ERROR), exception)

    def __await__(self) -> Generator[asyncio.Future[R], None, R]:
        return self.future.__await__()


@dataclass
class CommTask(Task[R]):
    """A task that can be communicated with."""

    comm: Comm
    """The communication object to communicate with the worker."""

    def on_update(
        self,
        f: Callable[[Self, Msg], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self, Msg], bool] | None = None,
    ) -> None:
        """Called when the task is finished."""
        name = name if name else f"{self.name}-update-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.UPDATE), f, name=name, pred=when)

    def on_waiting(
        self,
        f: Callable[[Self], None],
        *,
        name: CallbackName | None = None,
        when: Callable[[Self], bool] | None = None,
    ) -> None:
        """Called when the task is waiting to recieve something."""
        name = name if name else f"{self.name}-waiting-{str(uuid4())}"
        self.events.on((self.name, TaskStatus.WAITING), f, name=name, pred=when)

    async def _communicate(self) -> None:
        """Communicate with the task."""
        while True:
            # Try recieve a message from the worker
            # and emit an event once it occures
            try:
                msg = await self.comm.as_async.recv()
                logger.debug(f"Worker {self.name}: receieved {msg}")
                if msg == TaskStatus.WAITING:
                    self.events.emit((self.name, TaskStatus.WAITING), self)
                else:
                    self.events.emit((self.name, TaskStatus.UPDATE), self, msg)
            except EOFError:
                logger.debug(f"Worker {self.name}: closed connection")
                break

        # We are out of the loop, there's no way to communicate with
        # the worker anymore, close out and remove reference to this
        # task from the scheduler
        self.comm.close()

    @classmethod
    def from_task(cls, task: Task, comm: Comm) -> Self:
        """Create a CommTask from a Task."""
        return cls(name=task.name, future=task.future, events=task.events, comm=comm)
