"""A module containing the Comm class.

???+ note

    Please see the documentation for the [`Task`][amltk.scheduling.task.Task]
    for basics of a task.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import Pipe
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec, TypeAlias, override

from more_itertools import first_true

from amltk.asyncm import AsyncConnection
from amltk.events import Event
from amltk.scheduling.task_plugin import TaskPlugin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from typing_extensions import Self

    from amltk.scheduling.task import Task

    CommID: TypeAlias = int


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class Comm:
    """A communication channel between a worker and scheduler.

    For duplex connections, such as returned by python's builtin
    [`Pipe`][multiprocessing.Pipe], use the
    [`create(duplex=...)`][amltk.Comm.create] class method.

    Attributes:
        connection: The underlying Connection
        id: The id of the comm.
    """

    MESSAGE: Event[Comm.Msg] = Event("commtask-message")
    """A Task has sent a message."""

    REQUEST: Event[Comm.Msg] = Event("commtask-request")
    """A Task is waiting for a response."""

    CLOSE: Event[[]] = Event("commtask-close")
    """The task has signalled it's close."""

    def __init__(self, connection: Connection) -> None:
        """Initialize the Comm.

        Args:
            connection: The underlying Connection
        """
        super().__init__()
        self.connection = connection
        self.id: CommID = id(self)

    def send(self, obj: Any) -> None:
        """Send a message.

        Args:
            obj: The object to send.
        """
        try:
            self.connection.send(obj)
        except BrokenPipeError:
            # It's possble that the connection was closed by the other end
            # before we could send the message.
            logger.warning(f"Broken pipe error while sending message {obj}")

    def close(self, *, wait_for_ack: bool = False) -> None:
        """Close the connection.

        Args:
            wait_for_ack: If `True`, wait for an acknowledgement from the
                other end before closing the connection.
        """
        if not self.connection.closed:
            try:
                self.connection.send(Comm.Msg.Kind.CLOSE)
            except BrokenPipeError:
                # It's possble that the connection was closed by the other end
                # before we could close it.
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error sending close signal: {type(e)}{e}")

            if wait_for_ack:
                try:
                    logger.debug("Waiting for ACK")
                    self.connection.recv()
                    logger.debug("Recieved ACK")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error waiting for ACK: {type(e)}{e}")

            try:
                self.connection.close()
            except OSError:
                # It's possble that the connection was closed by the other end
                # before we could close it.
                pass
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error closing connection: {type(e)}{e}")

    @classmethod
    def create(cls, *, duplex: bool = True) -> tuple[Self, Self]:
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
    ) -> Comm.Msg | T:
        ...

    # No block with no default
    @overload
    def request(
        self,
        msg: Any | None = ...,
        *,
        block: Literal[False] | float,
        default: None = None,
    ) -> Comm.Msg | None:
        ...

    # Block
    @overload
    def request(
        self,
        msg: Any | None = ...,
        *,
        block: Literal[True] = True,
    ) -> Comm.Msg:
        ...

    def request(
        self,
        msg: Any | None = None,
        *,
        block: bool | float = True,
        default: T | None = None,
    ) -> Comm.Msg | T | None:
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
        self.send((Comm.Msg.Kind.REQUEST, msg))
        response = self.connection.poll(timeout=poll_timeout)
        return default if not response else self.connection.recv()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close(wait_for_ack=False)

    @dataclass
    class Msg(Generic[T]):
        """A message sent over a communication channel.

        Attributes:
            task: The task that sent the message.
            comm: The communication channel.
            future: The future of the task.
            data: The data sent by the task.
        """

        task: Task = field(repr=False)
        comm: Comm = field(repr=False)
        future: asyncio.Future = field(repr=False)
        data: T
        identifier: CommID

        def respond(self, response: Any) -> None:
            """Respond to the message.

            Args:
                response: The response to send back to the task.
            """
            self.comm.send(response)

        class Kind(Enum):
            """The kind of message."""

            CLOSE = auto()
            MESSAGE = auto()
            REQUEST = auto()

    class Plugin(TaskPlugin):
        """A plugin that handles communication with a worker."""

        name: ClassVar[str] = "comm-plugin"

        def __init__(
            self,
            create_comms: Callable[[], tuple[Comm, Comm]] | None = None,
        ) -> None:
            """Initialize the plugin.

            Args:
                create_comms: A function that creates a pair of communication
                    channels. Defaults to `Comm.create`.
            """
            super().__init__()
            if create_comms is None:
                create_comms = Comm.create

            self.create_comms = create_comms
            self.comms: dict[CommID, tuple[Comm, Comm]] = {}
            self.communication_tasks: dict[asyncio.Future, asyncio.Task] = {}
            self.task: Task

        @override
        def attach_task(self, task: Task) -> None:
            """Attach the plugin to a task.

            This method is called when the plugin is attached to a task. This
            is the place to subscribe to events on the task, create new subscribers
            for people to use or even store a reference to the task for later use.

            Args:
                task: The task the plugin is being attached to.
            """
            self.task = task
            task.on_submitted(self._establish_connection)

        @override
        def pre_submit(
            self,
            fn: Callable[P, R],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> tuple[Callable[P, R], tuple, dict] | None:
            """Pre-submit hook.

            This method is called before the task is submitted.

            Args:
                fn: The task function.
                *args: The arguments to the task function.
                **kwargs: The keyword arguments to the task function.

            Returns:
                A tuple of the task function, arguments and keyword arguments
                if the task should be submitted, or `None` if the task should
                not be submitted.
            """
            from amltk.optimization.trial import Trial

            host_comm, worker_comm = self.create_comms()
            # NOTE: This works but not sure why pyright is complaining

            trial = first_true(
                (a for a in args if isinstance(a, Trial)),
                default=None,
            )
            if trial is None:
                if "comm" in kwargs:
                    raise ValueError(
                        "Can't attach a comm as there is already a kwarg named `comm`.",
                    )
                kwargs.update({"comm": worker_comm})

            # We don't necessarily know if the future will be submitted. If so,
            # we will use this index later to retrieve the host_comm
            self.comms[worker_comm.id] = (host_comm, worker_comm)
            return fn, args, kwargs

        @override
        def copy(self) -> Self:
            """Return a copy of the plugin.

            Please see [`TaskPlugin.copy()`][amltk.TaskPlugin.copy].
            """
            return self.__class__(create_comms=self.create_comms)

        def _establish_connection(
            self,
            f: asyncio.Future,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            from amltk.optimization.trial import Trial

            trial = first_true(
                (a for a in args if isinstance(a, Trial)),
                default=None,
            )
            if trial is None:
                if "comm" not in kwargs:
                    raise ValueError(
                        "Cannot find a comm as there is no kwarg named `comm`.",
                        "and cannot find comm from a trial as there is no trial in"
                        " the arguments."
                        f"\nArgs: {args} kwargs: {kwargs}",
                    )
                worker_comm = kwargs["comm"]
            else:
                worker_comm = trial.plugins["comm"]

            host_comm, worker_comm = self.comms[worker_comm.id]
            self.communication_tasks[f] = asyncio.create_task(
                self._communicate(f, host_comm, worker_comm),
            )

        async def _communicate(
            self,
            future: asyncio.Future,
            host_comm: Comm,
            worker_comm: Comm,
        ) -> None:
            """Communicate with the task.

            This is a coroutine that will run until the scheduler is stopped or
            the comms have finished.
            """
            worker_id = worker_comm.id
            task_name = self.task.name
            name = f"{task_name}({worker_id})"

            while True:
                try:
                    data = await host_comm.as_async.request()
                    logger.debug(f"{self.name}: receieved {data=}")

                    # When we recieve CLOSE, the task has signalled it's
                    # close and we emit a CLOSE event. This should break out
                    # of the loop as we expect no more signals after this point
                    if data is Comm.Msg.Kind.CLOSE:
                        self.task.emit(Comm.CLOSE)
                        break

                    # When we recieve (REQUEST, data), this was sent with
                    # `request` and we emit a REQUEST event
                    if (
                        isinstance(data, tuple)
                        and len(data) == 2  # noqa: PLR2004
                        and data[0] == Comm.Msg.Kind.REQUEST
                    ):
                        _, real_data = data
                        msg = Comm.Msg(
                            self.task,
                            host_comm,
                            future,
                            real_data,
                            identifier=worker_id,
                        )
                        self.task.emit(Comm.REQUEST, msg)

                    # Otherwise it's just a simple `send` with some data we
                    # emit as a MESSAGE event
                    else:
                        msg = Comm.Msg(
                            self.task,
                            host_comm,
                            future,
                            data,
                            identifier=worker_id,
                        )
                        self.task.emit(Comm.MESSAGE, msg)

                except EOFError:
                    logger.debug(f"{name}: closed connection")
                    break

            logger.debug(f"{name}: finished communication, closing comms")

            # When the loop is finished, we can't communicate, close the comm
            # We explicitly don't wait for any acknowledgment from the worker
            host_comm.close(wait_for_ack=False)
            worker_comm.close()

            # Remove the reference to the work comm so it gets garbarged
            del self.comms[worker_id]


@dataclass
class AsyncComm:
    """A async wrapper of a Comm."""

    comm: Comm

    @overload
    async def request(
        self,
        *,
        timeout: float,
        default: None = None,
    ) -> Comm.Msg | None:
        ...

    @overload
    async def request(self, *, timeout: float, default: T) -> Comm.Msg | T:
        ...

    @overload
    async def request(self, *, timeout: None = None) -> Comm.Msg:
        ...

    async def request(
        self,
        *,
        timeout: float | None = None,
        default: T | None = None,
    ) -> Comm.Msg | T | None:
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

    async def send(self, obj: Comm.Msg) -> None:
        """Send a message.

        Args:
            obj: The message to send.
        """
        return await AsyncConnection(self.comm.connection).send(obj)
