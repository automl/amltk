"""A module containing the CommTask class.

???+ note
    Please see the documentation for the [`Task`][byop.scheduling.task.Task]
    for basics of a task.
"""
from __future__ import annotations

import asyncio
from asyncio import Future
from dataclasses import dataclass, field
import logging
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)

from typing_extensions import Self

from byop.asyncm import AsyncConnection
from byop.events import Event
from byop.scheduling.scheduler import Scheduler
from byop.scheduling.task import Task

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


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

    def send(self, obj: Any) -> None:
        """Send a message.

        Args:
            obj: The object to send.
        """
        self.connection.send(obj)

    def close(self) -> None:
        """Close the connection."""
        if not self.connection.closed:
            try:
                self.connection.send(CommTask.CLOSE)
            except BrokenPipeError:
                # It's possble that the connection was closed by the other end
                # before we could close it.
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error sending close signal: {type(e)}{e}")

            try:
                self.connection.close()
            except OSError:
                # It's possble that the connection was closed by the other end
                # before we could close it.
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error closing connection: {type(e)}{e}")

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
    def request(
        self,
        msg: Any | None = ...,
        *,
        block: Literal[False] | float,
        default: T,
    ) -> CommTask.Msg | T:
        ...

    # No block with no default
    @overload
    def request(
        self,
        msg: Any | None = ...,
        *,
        block: Literal[False] | float,
        default: None = None,
    ) -> CommTask.Msg | None:
        ...

    # Block
    @overload
    def request(
        self,
        msg: Any | None = ...,
        *,
        block: Literal[True] = True,
    ) -> CommTask.Msg:
        ...

    def request(
        self,
        msg: Any | None = None,
        *,
        block: bool | float = True,
        default: T | None = None,
    ) -> CommTask.Msg | T | None:
        """Receive a message.

        Args:
            msg: The message to send to the other end of the connection.
                If left empty, will be `None`.
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
        self.send((CommTask.REQUEST, msg))
        response = self.connection.poll(timeout=poll_timeout)
        return default if not response else self.connection.recv()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass
class AsyncComm:
    """A async wrapper of a Comm."""

    comm: Comm

    @overload
    async def request(
        self, *, timeout: float, default: None = None
    ) -> CommTask.Msg | None:
        ...

    @overload
    async def request(self, *, timeout: float, default: T) -> CommTask.Msg | T:
        ...

    @overload
    async def request(self, *, timeout: None = None) -> CommTask.Msg:
        ...

    async def request(
        self,
        *,
        timeout: float | None = None,
        default: T | None = None,
    ) -> CommTask.Msg | T | None:
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

    async def send(self, obj: CommTask.Msg) -> None:
        """Send a message.

        Args:
            obj: The message to send.
        """
        return await AsyncConnection(self.comm.connection).send(obj)


# TODO: Update this docstring
class CommTask(Task[Concatenate[Comm, P], R]):
    """A task that can communicate with a remote worker.

    An extended version of [`Task`][byop.scheduling.task.Task] which
    also provides a [`Comm`][byop.scheduling.comm_task.Comm] object to
    communicate with task once it's been dispatched.

    ```python
    # Define some function to run
    def calculate(comm: Comm, x: int) -> int:
        first_update = x * 2
        comm.send(first_update)  # (1)!

        second_update = x * 3
        comm.send(second_update)  # (2)!

        last_multiplier = comm.request()  # (3)!
        result = x * next_multiplier
        return result  # (4)!

    scheduler = Scheduler.with_processes(2)

    my_comm_task = scheduler.task("good-name", calculate, comms=True) # (5)!

    my_comm_task.on_message(lambda task, msg: print(msg)) # (6)!
    my_comm_task.on_waiting(lambda task: task.send(42)) # (7)!

    my_comm_task.on_return(lambda result: print(results))
    ```

    1. The task sends `x * 2` to the scheduler,
        triggering [`MESSAGE`][byop.scheduling.CommTask.MESSAGE].
    2. The task can repeat as many times as it wants
    3. The task blocks until it recieves a message from the scheduler,
        triggering [`REQUEST`][byop.scheduling.CommTask.REQUEST].
    4. The task returns a result, triggering
        [`DONE`][byop.scheduling.Task.DONE] and
        [`RETURNED`][byop.scheduling.Task.RETURNED].
    5. Create a task with a [`Comm`][byop.scheduling.Comm].
    6. Register a callback to be called when the task sends an update.
    7. Register a callback to be called when the task is waiting for a
        message from the scheduler.


    Attributes:
        name: The name of the task.
        function: The function of this task
        n_called: How many times this task has been called.
        call_limit: How many times this task can be run. Defaults to `None`
    """

    MESSAGE: Event[CommTask.Msg] = Event("commtask-message")
    """A Task has sent a message."""

    REQUEST: Event[CommTask.Msg] = Event("commtask-request")
    """A Task is waiting for a response."""

    CLOSE: Event[[]] = Event("commtask-close")
    """The task has signalled it's close."""

    def __init__(
        self,
        function: Callable[Concatenate[Comm, P], R],
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
        # NOTE: Mypy seems to be incorrect here:
        #   error: Argument 1 to "__init__" of "Task" has incompatible type
        #       "Callable[[Comm, **P], R]"; expected "Callable[[Comm, **P], R]"
        super().__init__(
            function,  # type: ignore
            scheduler,
            name=name,
            call_limit=call_limit,
            concurrent_limit=concurrent_limit,
            memory_limit=memory_limit,
            cpu_time_limit=cpu_time_limit,
            wall_time_limit=wall_time_limit,
        )

        # Holds the scheduler_comm, worker_comm, and communcation task
        # repsonsible for the communication
        # NOTE: It's important to hold a reference to the worker_comm so
        # it doesn get garbage collected and closed from the main process
        # so that the child process can use it
        self.comms: dict[Future, tuple[Comm, Comm, asyncio.Task]] = {}

        self.on_request = self.subscriber(self.REQUEST)
        self.on_message = self.subscriber(self.MESSAGE)

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R] | None:
        """Execute the task.

        This will create a [`Comm`][byop.scheduling.comm_task.Comm] object
        which can be used to communicate with the task and pass it as the first
        argument to the task function.

        Args:
            *args: The arguments to pass to the task function.
            **kwargs: The keyword arguments to pass to the task function.

        Returns:
            A future which can be used to get the result of the task and communicate
            with it. Will be `None` if the task has reached its call limit.
        """
        scheduler_comm, worker_comm = Comm.create(duplex=True)

        # NOTE: This works but not sure why pyright is complaining
        args = (worker_comm, *args)  # type: ignore
        task_future = super().__call__(*args, **kwargs)  # type: ignore
        if task_future is None:
            return None

        communcation_task = asyncio.create_task(self._communicate(task_future))
        self.comms[task_future] = (scheduler_comm, worker_comm, communcation_task)
        return task_future

    async def _communicate(self, future: Future[R]) -> None:
        """Communicate with the task.

        This is a coroutine that will run until the scheduler is stopped or
        the comms have finished.
        """
        comm, _, _ = self.comms[future]
        while True:
            try:
                data = await comm.as_async.request()
                logger.debug(f"{self.name}: receieved {data=}")

                # When we recieve (REQUEST, data), this was sent with
                # `request` and we emit a REQUEST event
                if (
                    isinstance(data, tuple)
                    and len(data) == 2  # noqa: PLR2004
                    and data[0] == CommTask.REQUEST
                ):
                    _, real_data = data
                    msg = CommTask.Msg(self, comm, future, real_data)
                    self.emit(CommTask.REQUEST, msg)

                # When we recieve CLOSE, the task has signalled it's
                # close and we emit a CLOSE event
                elif data == CommTask.CLOSE:
                    self.emit(CommTask.CLOSE)

                # Otherwise it's just a simple `send` with some data we
                # emit as a MESSAGE event
                else:
                    msg = CommTask.Msg(self, comm, future, data)
                    self.emit(CommTask.MESSAGE, msg)

            except EOFError:
                logger.debug(f"{self.name}: closed connection")
                break

    def _process_future(self, future: Future[R]) -> None:
        super()._process_future(future)
        comm, worker_comm, communcation_task = self.comms.pop(future)
        comm.close()
        worker_comm.close()
        communcation_task.cancel()

    @dataclass
    class Msg(Generic[T]):
        """A message sent over a communication channel.

        Attributes:
            task: The task that sent the message.
            comm: The communication channel.
            future: The future of the task.
            data: The data sent by the task.
        """

        task: CommTask
        comm: Comm = field(repr=False)
        future: Future = field(repr=False)
        data: T

        def respond(self, response: Any) -> None:
            """Respond to the message.

            Args:
                response: The response to send back to the task.
            """
            self.comm.send(response)
