"""A module containing the CommTask class.

???+ note
    Please see the documentation for the [`Task`][byop.scheduling.task.Task]
    for basics of a task.


"""
from __future__ import annotations

import asyncio
from asyncio import Future
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, TypeVar, overload

from typing_extensions import Self

from byop.asyncm import AsyncConnection
from byop.functional import funcname
from byop.scheduling.events import TaskEvent
from byop.scheduling.task import Task, TaskFuture
from byop.types import CallbackName, Msg, TaskParams, TaskReturn

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class CommTask(Task[TaskParams, TaskReturn]):
    """A task that can communicate with a remote worker.

    An extended version of [`Task`][byop.scheduling.task.Task] which
    also provides a [`Comm`][byop.scheduling.comm_task.Comm] object to
    communicate with task once it's been dispatched.

    All [events][byop.scheduling.events.TaskEvent] available, such as
    [`task.SUBMITTED`][byop.scheduling.task.Task.SUBMITTED] and
    [`task.DONE`][byop.scheduling.task.Task.DONE] are also available
    on this object.

    ```python
    # Define some function to run
    def calculate(comm: Comm, x: int) -> int:
        first_update = x * 2
        comm.send(first_update)  # (1)!

        second_update = x * 3
        comm.send(second_update)  # (2)!

        last_multiplier = comm.recv()  # (3)!
        result = x * next_multiplier
        return result  # (4)!

    scheduler = Scheduler.with_processes(2)

    my_comm_task = scheduler.task("good-name", calculate, comms=True) # (5)!

    my_comm_task.on_update(lambda task, msg: print(msg)) # (6)!
    my_comm_task.on_waiting(lambda task: task.send(42)) # (7)!

    my_comm_task.on_return(lambda result: print(results))
    ```

    1. The task sends `x * 2` to the scheduler,
        triggering [`UPDATE`][byop.scheduling.events.TaskEvent.UPDATE].
    2. The task can repeat as many times as it wants
    3. The task blocks until it recieves a message from the scheduler,
        triggering [`WAITING`][byop.scheduling.events.TaskEvent.WAITING].
    4. The task returns a result, triggering
        [`DONE`][byop.scheduling.events.TaskEvent.DONE] and
        [`RETURNED`][byop.scheduling.events.TaskEvent.RETURNED].
    5. Create a task with a [`Comm`][byop.scheduling.comm_task.Comm].
    6. Register a callback to be called when the task sends an update.
    7. Register a callback to be called when the task is waiting for a
        message from the scheduler.


    Attributes:
        name: The name of the task.
        function: The function of this task
        n_called: How many times this task has been called.
        call_limit: How many times this task can be run. Defaults to `None`
    """

    @overload
    def on(
        self,
        event: Literal[TaskEvent.SUBMITTED, TaskEvent.DONE, TaskEvent.CANCELLED],
        callback: Callable[[CommTaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.EXCEPTION],
        callback: Callable[[BaseException], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.RETURNED],
        callback: Callable[[TaskReturn], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.UPDATE],
        callback: Callable[[CommTaskFuture[TaskParams, TaskReturn], Msg], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    @overload
    def on(
        self,
        event: Literal[TaskEvent.WAITING],
        callback: Callable[[CommTaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = ...,
        when: Callable[[Counter[TaskEvent]], bool] | None = ...,
    ) -> Self:
        ...

    def on(
        self,
        event: TaskEvent,
        callback: Callable | Task,
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task emits an event.

        Args:
            event: The event to listen to.
            callback: The callback to call.
            name: A specific name to give the callback
            when: A function that takes the current state of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.
        """
        if name is None:
            name = callback.name if isinstance(callback, Task) else funcname(callback)
            name = f"on-{self.name}-{event}-{name}"

        pred = None if when is None else (lambda counts=self.counts: when(counts))
        _event = (self.name, event)
        self.scheduler.event_manager.on(_event, callback, name=name, when=pred)
        return self

    def on_update(
        self,
        callback: Callable[[CommTaskFuture[TaskParams, TaskReturn], Msg], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task sends an update.

        This is triggered by a task while it's running, using the
        [`send`][byop.scheduling.comm_task.Comm.send] method of its
        [`Comm`][byop.scheduling.comm_task.Comm].

        Args:
            callback: The callback to call. Must accept the task future as the
                first argument.
            name: The name of the callback.
            when: A function that takes the event counts of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.

        Returns:
            The task.
        """
        return self.on(TaskEvent.UPDATE, callback, name=name, when=when)

    def on_waiting(
        self,
        callback: Callable[[CommTaskFuture[TaskParams, TaskReturn]], Any],
        *,
        name: CallbackName | None = None,
        when: Callable[[Counter[TaskEvent]], bool] | None = None,
    ) -> Self:
        """Register a callback to be called when the task is waiting.

        This is triggered by a task while it's running, using the
        [`recv`][byop.scheduling.comm_task.Comm.send] method of its
        [`Comm`][byop.scheduling.comm_task.Comm]. It will block until a response
        has been recieved.

        Args:
            callback: The callback to call. Must accept the task future as the
                first argument.
            name: The name of the callback.
            when: A function that takes the event counts of the task and
                returns a boolean. If the boolean is True, the callback will
                be called.

        Returns:
            The task.
        """
        return self.on(TaskEvent.WAITING, callback, name=name, when=when)

    UPDATE: ClassVar = TaskEvent.UPDATE
    """An event triggered when a task has sent something with `send`."""

    WAITING: ClassVar = TaskEvent.WAITING
    """An event triggered when a task is waiting for a response."""


@dataclass(frozen=True)
class CommTaskFuture(TaskFuture[TaskParams, TaskReturn]):
    """A thin wrapper for a future and comm, with reference to the task description.

    This object will be passed to callbacks registered with
    [`on_update`][byop.scheduling.comm_task.CommTask.on_update] and
    [`on_waiting`][byop.scheduling.comm_task.CommTask.on_waiting]. It
    will allow you to send messages back to the task with
    [`send`][byop.scheduling.comm_task.Comm.send].

    Attributes:
        task: The task associated with this future.
        future: The future associated with this task.
        comm: The comm associated with this task.
    """

    desc: CommTask[TaskParams, TaskReturn]
    future: Future[TaskReturn] = field(repr=False)
    comm: Comm = field(repr=False)

    def send(self, msg: Msg) -> None:
        """Send a message to the worker.

        Args:
            msg: The message to send.
        """
        self.comm.send(msg)


@dataclass
class Comm:
    """A communication channel between a worker and scheduler.

    For duplex connections, such as returned by python's builtin
    [`Pipe`][multiprocessing.Pipe], use the
    [`create(duplex=...)`][byop.scheduling.comm_task.Comm.create] class method.

    Attributes:
        connection: The underlying Connection
    """

    connection: Connection

    def send(self, obj: Msg) -> None:
        """Send a message.

        Args:
            obj: The object to send.
        """
        self.connection.send(obj)

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    @classmethod
    def create(cls, *, duplex: bool = False) -> tuple[Self, Self]:
        """Create a pair of communication channels.

        Wraps the output of
        [`multiprocessing.Pipe(duplex=duplex)`][multiprocessing.Pipe].

        Args:
            duplex: Whether to allow for two-way communication

        Returns:
            A pair of communication channels.
        """
        reader, writer = Pipe(duplex=duplex)
        return cls(reader), cls(writer)

    @property
    def as_async(self) -> AsyncComm:
        """Return an async version of this comm."""
        return AsyncComm(self)

    # No block with a default
    @overload
    def recv(self, *, block: Literal[False] | float, default: T) -> Msg | T:
        ...

    # No block with no default
    @overload
    def recv(
        self, *, block: Literal[False] | float, default: None = None
    ) -> Msg | None:
        ...

    # Block
    @overload
    def recv(self, *, block: Literal[True] = True) -> Msg:
        ...

    def recv(
        self,
        *,
        block: bool | float = True,
        default: T | None = None,
    ) -> Msg | T | None:
        """Receive a message.

        Args:
            block: Whether to block until a message is received. If False, return
                default.
            default: The default value to return if block is False and no message
                is received. Defaults to None.

        Returns:
            The received message or the default.
        """
        if block is False:
            response = self.connection.poll()  # Non blocking poll
            return default if not response else self.connection.recv()

        # None indicates blocking poll
        poll_timeout = None if block is True else block
        self.send(TaskEvent.WAITING)
        response = self.connection.poll(timeout=poll_timeout)
        return default if not response else self.connection.recv()


@dataclass
class AsyncComm:
    """A async wrapper of a Comm."""

    comm: Comm

    async def recv(
        self,
        *,
        timeout: float | None = None,
        default: T | None = None,
    ) -> Msg | T:
        """Recieve a message.

        Args:
            timeout: The timeout in seconds to wait for a message.
            default: The default value to return if the timeout is reached.

        Returns:
            The message from the worker or the default value.
        """
        connection = AsyncConnection(self.comm.connection)
        result = await asyncio.wait_for(connection.recv(), timeout=timeout)
        return default if result is None else result

    async def send(self, obj: Msg) -> None:
        """Send a message.

        Args:
            obj: The message to send.
        """
        return await AsyncConnection(self.comm.connection).send(obj)
