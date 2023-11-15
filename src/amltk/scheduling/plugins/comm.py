"""The [`Comm.Plugin`][amltk.scheduling.plugins.comm.Comm.Plugin] enables
two way-communication with running [`Task`][amltk.scheduling.task.Task].

The [`Comm`][amltk.scheduling.plugins.comm.Comm] provides an easy interface to
communicate while the [`Comm.Msg`][amltk.scheduling.plugins.comm.Comm.Msg] encapsulates
messages between the main process and the `Task`.

??? tip "Usage"

    To setup a `Task` to work with a `Comm`, the `Task` **must accept a `comm` as
    it's first argument**.

    ```python exec="true" source="material-block" result="python" hl_lines="4-7 10 17-19 21-23"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def powers_of_two(comm: Comm, start: int, n: int) -> None:
        with comm.open():
            for i in range(n):
                comm.send(start ** (i+1))
    from amltk._doc import make_picklable; make_picklable(powers_of_two)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(powers_of_two, plugins=Comm.Plugin())
    results = []

    @scheduler.on_start
    def on_start():
        task.submit(2, 5)

    @task.on("comm-open")
    def on_open(msg: Comm.Msg):
        print(f"Task has opened | {msg}")

    @task.on("comm-message")
    def on_message(msg: Comm.Msg):
        results.append(msg.data)

    scheduler.run()
    print(results)
    ```

    You can also block a worker, waiting for a response from the main process, allowing for the
    worker to [`request()`][amltk.scheduling.plugins.comm.Comm.request] data from the main process.

    ```python exec="true" source="material-block" result="python" hl_lines="7 20-23"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def my_worker(comm: Comm, n_tasks: int) -> None:
        with comm.open():
            for task_number in range(n_tasks):
                task = comm.request(task_number)
                comm.send(f"Task recieved {task} for {task_number}")
    from amltk._doc import make_picklable; make_picklable(my_worker)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(my_worker, plugins=Comm.Plugin())

    items = ["A", "B", "C"]
    results = []

    @scheduler.on_start
    def on_start():
        task.submit(n_tasks=3)

    @task.on("comm-request")
    def on_request(msg: Comm.Msg):
        task_number = msg.data
        msg.respond(items[task_number])

    @task.on("comm-message")
    def on_message(msg: Comm.Msg):
        results.append(msg.data)

    scheduler.run()
    print(results)
    ```

??? example "`@events`"

    === "`@comm-message`"

        ::: amltk.scheduling.plugins.comm.Comm.MESSAGE

    === "`@comm-request`"

        ::: amltk.scheduling.plugins.comm.Comm.REQUEST

    === "`@comm-open`"

        ::: amltk.scheduling.plugins.comm.Comm.OPEN

    === "`@comm-close`"

        ::: amltk.scheduling.plugins.comm.Comm.CLOSE

??? warning "Supported Backends"

    The current implementation relies on [`Pipe`][multiprocessing.Pipe] which only
    works between processes on the same system/cluster. There is also limited support
    with `dask` backends.

    This could be extended to allow for web sockets or other forms of connections
    but requires time. Please let us know in the Github issues if this is something
    you are interested in!
"""  # noqa: E501
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import (
    Pipe,
    TimeoutError as MPTimeoutError,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeAlias,
    TypeVar,
)
from typing_extensions import ParamSpec, override

from amltk._asyncm import AsyncConnection
from amltk.scheduling.events import Event
from amltk.scheduling.plugins.plugin import Plugin as TaskPlugin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from typing_extensions import Self

    from rich.panel import Panel

    from amltk.scheduling.task import Task

    CommID: TypeAlias = int


T = TypeVar("T")
M = TypeVar("M")
P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class AsyncComm:
    """A async wrapper of a Comm."""

    comm: Comm

    async def request(
        self,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Recieve a message.

        Args:
            timeout: The timeout in seconds to wait for a message, raises
                a [`Comm.TimeoutError`][amltk.scheduling.plugins.comm.Comm.TimeoutError]
                if the timeout is reached.
                If `None`, will wait forever.

        Returns:
            The message from the worker or the default value.
        """
        connection = AsyncConnection(self.comm.connection)
        try:
            return await asyncio.wait_for(connection.recv(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise Comm.TimeoutError(
                f"Timed out waiting for response from {self.comm}",
            ) from e

    async def send(self, obj: Any) -> None:
        """Send a message.

        Args:
            obj: The message to send.
        """
        return await AsyncConnection(self.comm.connection).send(obj)


class Comm:
    """A communication channel between a worker and scheduler.

    For duplex connections, such as returned by python's builtin
    [`Pipe`][multiprocessing.Pipe], use the
    [`create(duplex=...)`][amltk.Comm.create] class method.

    Adds three new events to the task:

    * [`@comm-message`][amltk.scheduling.plugins.comm.Comm.MESSAGE]
    * [`@comm-request`][amltk.scheduling.plugins.comm.Comm.REQUEST]
    * [`@comm-close`][amltk.scheduling.plugins.comm.Comm.CLOSE]
    * [`@comm-open`][amltk.scheduling.plugins.comm.Comm.OPEN]

    Attributes:
        connection: The underlying Connection
        id: The id of the comm.
    """

    MESSAGE: Event[Comm.Msg] = Event("comm-message")
    """A Task has sent a message to the main process.

    ```python exec="true" source="material-block" html="true" hl_lines="6 11-13"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def fn(comm: Comm, x: int) -> int:
        with comm.open():
            comm.send(x + 1)

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(fn, plugins=Comm.Plugin())

    @task.on("comm-message")
    def callback(msg: Comm.Msg):
        print(msg.data)
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    REQUEST: Event[Comm.Msg] = Event("comm-request")
    """A Task has sent a request.

    ```python exec="true" source="material-block" html="true" hl_lines="6 16-18"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def greeter(comm: Comm, greeting: str) -> None:
        with comm.open():
            name = comm.request()
            comm.send(f"{greeting} {name}!")
    from amltk._doc import make_picklable; make_picklable(greeter)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(greeter, plugins=Comm.Plugin())

    @scheduler.on_start
    def on_start():
        task.submit("Hello")

    @task.on("comm-request")
    def on_request(msg: Comm.Msg):
        msg.respond("Alice")

    @task.on("comm-message")
    def on_msg(msg: Comm.Msg):
        print(msg.data)

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """  # noqa: E501

    OPEN: Event[Comm.Msg] = Event("comm-open")
    """The task has signalled it's open.

    ```python exec="true" source="material-block" html="true" hl_lines="5 15-17"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def fn(comm: Comm) -> None:
        with comm.open():
            pass
   from amltk._doc import make_picklable; make_picklable(fn)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(fn, plugins=Comm.Plugin())

    @scheduler.on_start
    def on_start():
        task.submit()

    @task.on("comm-open")
    def callback(msg: Comm.Msg):
        print("Comm has just used comm.open()")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    CLOSE: Event[Comm.Msg] = Event("comm-close")
    """The task has signalled it's close.

    ```python exec="true" source="material-block" html="true" hl_lines="7 17-19"
    from amltk.scheduling import Scheduler
    from amltk.scheduling.plugins import Comm

    def fn(comm: Comm) -> None:
        with comm.open():
            pass
            # Will send a close signal to the main process as it exists this block

        print("Done")
    from amltk._doc import make_picklable; make_picklable(fn)  # markdown-exec: hide
    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(fn, plugins=Comm.Plugin())

    @scheduler.on_start
    def on_start():
        task.submit()

    @task.on("comm-close")
    def on_close(msg: Comm.msg):
        print(f"Worker close with {msg}")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
    ```
    """

    def __init__(self, connection: Connection) -> None:
        """Initialize the Comm.

        Args:
            connection: The underlying Connection
        """
        super().__init__()
        self.connection = connection
        self.id: CommID = id(self)

    def _send_pipe(self, obj: Any) -> None:
        self.connection.send(obj)

    def send(self, obj: Any) -> None:
        """Send a message.

        Args:
            obj: The object to send.
        """
        self._send_pipe((Comm.Msg.Kind.MESSAGE, obj))

    def close(  # noqa: PLR0912, C901
        self,
        msg: Any | None = None,
        *,
        wait_for_ack: bool = False,
        okay_if_broken_pipe: bool = False,
        side: str = "",
    ) -> None:
        """Close the connection.

        Args:
            msg: The message to send to the other end of the connection.
            wait_for_ack: If `True`, wait for an acknowledgement from the
                other end before closing the connection.
            okay_if_broken_pipe: If `True`, will not log an error if the
                connection is already closed.
            side: The side of the connection for naming purposes.
        """
        if not self.connection.closed:
            kind = Comm.Msg.Kind.CLOSE_WITH_ACK if wait_for_ack else Comm.Msg.Kind.CLOSE
            try:
                self._send_pipe((kind, msg))
            except BrokenPipeError as e:
                if not okay_if_broken_pipe:
                    logger.error(f"{side} - Error sending close signal: {type(e)}{e}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"{side} - Error sending close signal: {type(e)}{e}")
            else:
                if wait_for_ack:
                    logger.debug(f"{side} - Waiting for ACK")
                    try:
                        recieved_msg = self.connection.recv()
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            f"{side} - Error waiting for ACK, closing: {type(e)}{e}",
                        )
                    else:
                        match recieved_msg:
                            case Comm.Msg.Kind.WORKER_CLOSE_REQUEST:
                                logger.error(
                                    f"{side} - Worker recieved request to close!",
                                )
                            case Comm.Msg.Kind.ACK:
                                logger.debug(f"{side} - Recieved ACK, closing")
                            case _:
                                logger.warning(
                                    f"{side} - Expected ACK but {recieved_msg=}",
                                )
            finally:
                try:
                    self.connection.close()
                except OSError:
                    # It's possble that the connection was closed by the other end
                    # before we could close it.
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.error(f"{side} - Error closing connection: {type(e)}{e}")

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

    def request(
        self,
        msg: Any | None = None,
        *,
        timeout: None | float = None,
    ) -> Any:
        """Receive a message.

        Args:
            msg: The message to send to the other end of the connection.
                If left empty, will be `None`.
            timeout: If float, will wait for that many seconds, raising an exception
                if exceeded. Otherwise, None will wait forever.

        Raises:
            Comm.TimeoutError: If the timeout is reached.
            Comm.CloseRequestError: If the other end needs to abruptly end and
                can not fufill the request. If thise error is thrown, the worker
                should finish as soon as possible.

        Returns:
            The received message or the default.
        """
        self._send_pipe((Comm.Msg.Kind.REQUEST, msg))
        if not self.connection.poll(timeout):
            raise Comm.TimeoutError(f"Timed out waiting for response for {msg}")

        response = self.connection.recv()
        if response == Comm.Msg.Kind.WORKER_CLOSE_REQUEST:
            logger.error("Worker recieved request to close!")
            raise Comm.CloseRequestError()

        return response

    @contextmanager
    def open(
        self,
        opening_msg: Any | None = None,
        *,
        wait_for_ack: bool = False,
        side: str = "worker",
    ) -> Iterator[Self]:
        """Open the connection.

        Args:
            opening_msg: The message to send to the main process
                when the connection is opened.
            wait_for_ack: If `True`, wait for an acknowledgement from the
                other end before closing the connection and exiting the
                context manager.
            side: The side of the connection for naming purposes.
                Usually this is only done on the `"worker"` side.

        Yields:
            The comm.
        """
        self._send_pipe((Comm.Msg.Kind.OPEN, opening_msg))
        yield self
        self.close(wait_for_ack=wait_for_ack, side=side)

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
            self.communication_tasks: list[asyncio.Task] = []
            self.task: Task
            self.open_comms: set[CommID] = set()

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
            task.emitter.add_event(Comm.MESSAGE, Comm.REQUEST, Comm.OPEN, Comm.CLOSE)
            task.on_submitted(self._begin_listening, hidden=True)

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
            host_comm, worker_comm = self.create_comms()

            # We don't necessarily know if the future will be submitted. If so,
            # we will use this index later to retrieve the host_comm
            self.comms[worker_comm.id] = (host_comm, worker_comm)

            # Make sure to include the Comm
            return fn, (worker_comm, *args), kwargs

        @override
        def copy(self) -> Self:
            """Return a copy of the plugin.

            Please see [`Plugin.copy()`][amltk.scheduling.Plugin.copy].
            """
            return self.__class__(create_comms=self.create_comms)

        def _begin_listening(self, f: asyncio.Future, *args: Any, **_: Any) -> Any:
            match args:
                case (worker_comm, *_) if isinstance(worker_comm, Comm):
                    worker_comm = args[0]
                case _:
                    raise ValueError(f"Expected first arg to be a Comm, got {args[0]}")

            host_comm, worker_comm = self.comms[worker_comm.id]

            coroutine = asyncio.create_task(
                self._communicate(f, host_comm, worker_comm),
            )
            coroutine.add_done_callback(self._deregister_comm_coroutine)

            # NOTE: Asyncio coroutines must have a reference stored somewhere so
            # we need to hold on to it until it's done.
            self.communication_tasks.append(coroutine)

        def _deregister_comm_coroutine(self, coroutine: asyncio.Task) -> None:
            if coroutine in self.communication_tasks:
                self.communication_tasks.remove(coroutine)
            else:
                logger.warning(f"Communication coroutine {coroutine} not found!")

            if (exception := coroutine.exception()) is not None:
                raise exception

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
            task_name = self.task.unique_ref
            name = f"Task [{task_name}] (worker_id: {worker_id})"
            closed = False

            try:
                while not closed and (_msg := await host_comm.as_async.request()):
                    assert isinstance(_msg, tuple), "Expected (msg_kind, data)!"
                    msg_kind, data = _msg
                    logger.debug(f"{self.name}: receieved {msg_kind} with {data=}")

                    match msg_kind:
                        # Other side has closed the connection, break out of coroutine
                        case Comm.Msg.Kind.CLOSE:
                            closed = True
                        case Comm.Msg.Kind.CLOSE_WITH_ACK:
                            host_comm._send_pipe(Comm.Msg.Kind.ACK)
                            closed = True
                        case Comm.Msg.Kind.OPEN:
                            self.open_comms.add(worker_id)
                        case _:
                            pass

                    event = EVENT_LOOKUP[msg_kind]
                    msg = Comm.Msg(
                        comm=host_comm,
                        data=data,
                        kind=msg_kind,
                        future=future,
                        task=self.task,
                    )
                    self.task.emitter.emit(event, msg)

            except EOFError:
                # This means the connection dropped to the worker, however this is not
                # an error in the main process and so we can safely ignore that.
                logger.debug(f"{name}: closed connection")
            except Exception as e:
                # Something unexpected happened in the main process, either from us or
                # from a users callback. In this case we want to raise the exception
                logger.error(
                    f"{name}: Exception occured in scheduler or callbacks!",
                    exc_info=e,
                )

                # NOTE: It's important that we let the worker know that something went
                # wrong, especially if it's requesting things. The worker will only
                # see this msg when it does a `request()`
                host_comm._send_pipe(Comm.Msg.Kind.WORKER_CLOSE_REQUEST)
                raise e
            finally:
                # Make sure we do all the clean up!
                logger.debug(f"{name}: finished communication, closing comms")

                # We don't necessarily know how we got here but
                host_comm.close(
                    wait_for_ack=False,
                    okay_if_broken_pipe=True,
                    side="host",
                )
                worker_comm.close(
                    wait_for_ack=False,
                    okay_if_broken_pipe=True,
                    side="host-on-worker-comm",
                )

                if worker_id in self.open_comms:
                    self.open_comms.remove(worker_id)

                # Remove the reference to the work comm so it gets garbarged
                del self.comms[worker_id]
                logger.debug(f"{name}: finished and cleaned")

        @override
        def __rich__(self) -> Panel:
            from rich.panel import Panel
            from rich.text import Text

            return Panel(
                Text("Open Connections: ").append(str(len(self.open_comms)), "yellow"),
                title=f"Plugin {self.name}",
            )

    @dataclass
    class Msg(Generic[T]):
        """A message sent over a communication channel.

        Attributes:
            task: The task that sent the message.
            comm: The communication channel.
            future: The future of the task.
            data: The data sent by the task.
        """

        kind: Kind
        data: T
        comm: Comm = field(repr=False)
        future: asyncio.Future = field(repr=False)
        task: Task = field(repr=False)

        def respond(self, response: Any) -> None:
            """Respond to the message.

            Args:
                response: The response to send back to the task.
            """
            self.comm._send_pipe(response)

        class Kind(str, Enum):
            """The kind of message."""

            CLOSE = "close"
            CLOSE_WITH_ACK = "close-with-ack"
            WORKER_CLOSE_REQUEST = "worker-close-request"
            OPEN = "open"
            MESSAGE = "message"
            REQUEST = "request"
            ACK = "ack"

            @override
            def __str__(self) -> str:
                return self.value

    class TimeoutError(MPTimeoutError):  # noqa: A001
        """A timeout error for communications."""

    class CloseRequestError(RuntimeError):
        """An exception happened in the main process and it send
        a response to the worker to raise this exception.
        """


EVENT_LOOKUP = {
    Comm.Msg.Kind.CLOSE: Comm.CLOSE,
    Comm.Msg.Kind.CLOSE_WITH_ACK: Comm.CLOSE,
    Comm.Msg.Kind.OPEN: Comm.OPEN,
    Comm.Msg.Kind.MESSAGE: Comm.MESSAGE,
    Comm.Msg.Kind.REQUEST: Comm.REQUEST,
}
