"""A module defining communication channels between scheduler and workers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any, Literal, TypeVar, overload

from typing_extensions import Self

from byop.asyncm import AsyncConnection
from byop.scheduling.events import TaskEvent

T = TypeVar("T")


@dataclass
class Comm:
    """A communication channel between a worker and scheduler."""

    connection: Connection

    def send(self, obj: Any) -> None:
        """Send a message."""
        self.connection.send(obj)

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    # No block with a default
    @overload
    def recv(self, *, block: Literal[False] | float, default: T) -> Any | T:
        ...

    # No block with no default
    @overload
    def recv(
        self, *, block: Literal[False] | float, default: None = None
    ) -> Any | None:
        ...

    # Block
    @overload
    def recv(self, *, block: Literal[True] = True) -> Any:
        ...

    def recv(
        self,
        *,
        block: bool | float = True,
        default: T | None = None,
    ) -> Any | T | None:
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

    @property
    def as_async(self) -> AsyncComm:
        """Return an async version of this comm."""
        return AsyncComm(self)

    @classmethod
    def create(cls, *, duplex: bool = False) -> tuple[Self, Self]:
        """TrueCreate a pair of communication channels.

        Wraps the output of `multiprocessing.Pipe(duplex=duplex)`.

        Args:
            duplex: Whether to create a pair of duplex channels.

        Returns:
            A pair of communication channels.
        """
        reader, writer = Pipe(duplex=duplex)
        return cls(reader), cls(writer)


@dataclass
class AsyncComm:
    """A async wrapper of a Comm."""

    comm: Comm

    async def recv(
        self,
        *,
        timeout: float | None = None,
        default: T | None = None,
    ) -> Any | T:
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

    async def send(self, obj: Any) -> None:
        """Send a message.

        Args:
            obj: The message to send.
        """
        return await AsyncConnection(self.comm.connection).send(obj)
