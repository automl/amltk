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
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast
from uuid import uuid4

from attrs import evolve, field, frozen
from typing_extensions import Self

from byop.event_manager import EventManager
from byop.scheduling.comm import Comm
from byop.scheduling.events import TaskEvent
from byop.types import CallbackName, Msg, TaskName

R = TypeVar("R")


@frozen(kw_only=True)
class TaskDescription(Generic[R]):
    """A task is a unit of work that can be scheduled by the scheduler.

    Note:
        The `on_<event>` methods can take three forms of predicates,
        `every`, `count`, and `when`. An `all()` is used to combine
        them such that the handler will only be called if all of them
        evaluate to `True`.
    """

    event: ClassVar[type[TaskEvent]] = TaskEvent
    """The possible events a task can emit"""

    name: TaskName
    """The name of the task"""

    event_manager: EventManager = field(repr=False)
    """The eventmanager it will emit to."""

    f: Callable[..., R]
    args: tuple[Any, ...] = field(default=())
    kwargs: dict[str, Any] = field(default={})

    dispatch_f: Callable[[Self], Task[R]] = field(repr=False)
    """This function will be defined by the scheduler.

    It allows for delayed dispatch of task descriptions, for example if the user
    wants to recieve a task description, subscribe some events, and then dispatch
    it.
    """

    def on_submit(
        self,
        f: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when the task is submitted to the scheduler.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-submitted-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[Task[R]], Any], f)

        name = name if name else f"{self.name}-submitted-{str(uuid4())}"
        event = (self.name, TaskEvent.SUBMITTED)
        self.event_manager.on(event, f, name=name, every=every, count=count, when=when)
        return self

    def on_finish(
        self,
        f: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when the task is finished.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-submitted-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[Task[R]], Any], f)

        name = name if name else f"{self.name}-finish-{str(uuid4())}"
        event = (self.name, TaskEvent.FINISHED)
        self.event_manager.on(event, f, name=name, every=every, count=count, when=when)
        return self

    def on_success(
        self,
        f: Callable[[R], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[R], bool] | None = None,
    ) -> Self:
        """Called when the task is successfully completed.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-success-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[R], Any], f)

        name = name if name else f"{self.name}-success-{str(uuid4())}"
        event = (self.name, TaskEvent.SUCCESS)
        self.event_manager.on(event, f, name=name, every=every, count=count, when=when)
        return self

    def on_error(
        self,
        f: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[BaseException], bool] | None = None,
    ) -> Self:
        """Called when the task is finished but errored.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-error-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[BaseException], Any], f)

        name = name if name else f"{self.name}-error-{str(uuid4())}"
        event = (self.name, TaskEvent.ERROR)
        self.event_manager.on(event, f, name=name, every=every, count=count, when=when)
        return self

    def on_cancelled(
        self,
        f: Callable[[Task[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[Task[R]], bool] | None = None,
    ) -> Self:
        """Called when the task is cancelled before finishng.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-cancelled-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[Task[R]], Any], f)

        name = name if name else f"{self.name}-cancelled-{str(uuid4())}"
        event = (self.name, TaskEvent.CANCELLED)
        self.event_manager.on(event, f, name=name, every=every, count=count, when=when)
        return self

    @property
    def event_counts(self) -> dict[TaskEvent, int]:
        """Get the number of event counts for this task."""
        return {
            event: self.event_manager.count[(self.name, event)]
            for event in list(TaskEvent)
        }

    def __call__(self: Self, *args: Any, **kwargs: Any) -> Task[R]:  # noqa: ARG002
        """Dispatch this task.

        Note:
            This is mainly for convenience within the scheduler. Where possible,
            prefer to use the explicit `dispatch` method.

        Returns:
            A Task with the actual future attached
        """
        return self.dispatch_f(self)

    def dispatch(self: Self) -> Task[R]:
        """Dispatch this task.

        Returns:
            A Task with the actual future attached
        """
        return self.dispatch_f(self)

    def modified(self: Self, *args: Any, **kwargs: Any) -> Self:
        """Modify this task.

        Args:
            *args: The args to use
            **kwargs: The kwargs to use

        Returns:
            A new task with the modified attributes
        """
        return evolve(self, args=args, kwargs=kwargs)


class CommTaskDescription(TaskDescription[R]):
    """A task that can communicate with a remote worker.

    Note:
        The `on_<event>` methods can take three forms of predicates,
        `every`, `count`, and `when`. An `all()` is used to combine
        them such that the handler will only be called if all of them
        evaluate to `True`.
    """

    event: ClassVar[type[TaskEvent]] = TaskEvent
    """The possible events a task can emit"""

    name: TaskName
    """The name of the task"""

    event_manager: EventManager = field(repr=False)
    """The eventmanager it will emit to."""

    f: Callable[..., R]
    args: tuple[Any, ...] = field(default=())
    kwargs: dict[str, Any] = field(default={})

    dispatch_f: Callable[[Self], CommTask[R]] = field(repr=False)
    """This function will be defined by the scheduler.

    It allows for delayed dispatch of task descriptions, for example if the user
    wants to recieve a task description, subscribe some events, and then dispatch
    it.
    """

    def on_update(
        self,
        f: Callable[[CommTask[R], Msg], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[CommTask[R], Msg], bool] | None = None,
    ) -> Self:
        """Called when the task sends an update with `Comm.send`.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-update-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[CommTask[R], Msg], Any], f)

        name = name if name else f"{self.name}-update-{str(uuid4())}"
        event = (self.name, TaskEvent.UPDATE)
        self.event_manager.on(event, f, name=name, when=when, count=count, every=every)
        return self

    def on_waiting(
        self,
        f: Callable[[CommTask[R]], Any],
        *,
        name: CallbackName | None = None,
        every: int | None = None,
        count: Callable[[int], bool] | None = None,
        when: Callable[[CommTask[R]], bool] | None = None,
    ) -> Self:
        """Called when the task is waiting to recieve something with `Comm.recv`.

        Args:
            f: The function to call
            name: The name of the callback. If not provided, a random one will be
                generated with "{task-name}-waiting-{uuid4}".
            every: Call the handler when the event count is a multiple of `every` times.
            count: A callback the recieves the count of how many times this event has
                been emitted. If it returns `True`, the handler will be called.
            when: A callback that recieves the task. If it returns `True`, the
                handler will be called.

        Returns:
            Self
        """
        if isinstance(f, TaskDescription):
            name = name if name else f.name
            f = cast(Callable[[CommTask[R]], Any], f)

        name = name if name else f"{self.name}-waiting-{str(uuid4())}"
        event = (self.name, TaskEvent.WAITING)
        self.event_manager.on(event, f, name=name, when=when, count=count, every=every)
        return self

    def __call__(self: Self, *args: Any, **kwargs: Any) -> CommTask[R]:  # noqa: ARG002
        """Dispatch this task.

        Note:
            This is mainly for convenience within the scheduler. Where possible,
            prefer to use the explicit `dispatch` method.

        Returns:
            A Task with the actual future attached
        """
        return self.dispatch_f(self)

    def dispatch(self: Self) -> CommTask[R]:
        """Dispatch this task.

        Returns:
            A Task with the actual future attached
        """
        return self.dispatch_f(self)


@frozen(kw_only=True)
class Task(Generic[R]):
    """A thin wrapper for a future with a name and reference to the task description."""

    future: asyncio.Future[R] = field(repr=False)
    """The future holding the result"""

    desc: TaskDescription
    """The task description this is attributed to"""

    @property
    def name(self) -> TaskName:
        """The name of the task."""
        return self.desc.name

    def result(self) -> R:
        """Get the result of the task."""
        return self.future.result()

    def cancel(self) -> None:
        """Cancel the task."""
        self.future.cancel()

    def done(self) -> bool:
        """Check if the task is done."""
        return self.future.done()

    def cancelled(self) -> bool:
        """Check if the task is cancelled."""
        return self.future.cancelled()

    def exception(self) -> BaseException | None:
        """Get the exception of the task."""
        return self.future.exception()

    def modified(self, *args: Any, **kwargs: Any) -> TaskDescription:
        """Modify the task description.

        Returns:
            A new task description with the same name and function, but with the
            new args and kwargs.
        """
        return self.desc.modified(*args, **kwargs)


@frozen(kw_only=True)
class CommTask(Task[R]):
    """A thin wrapper for a future with a name and reference to the task description."""

    future: asyncio.Future[R] = field(repr=False)
    """The future holding the result"""

    desc: TaskDescription
    """The task description this is attributed to"""

    comm: Comm = field(repr=False)
    """The communication object to communicate with the worker."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch the task."""
        return self.desc.__call__(*args, **kwargs)

    def send(self, msg: Msg) -> None:
        """Send a message to the worker."""
        self.comm.send(msg)

    def recv(self) -> Msg:
        """Receive a message from the worker."""
        return self.comm.recv()

    @classmethod
    def from_task(cls, task: Task[R], comm: Comm) -> CommTask[R]:
        """Create a CommTask from a Task."""
        return cls(future=task.future, desc=task.desc, comm=comm)
