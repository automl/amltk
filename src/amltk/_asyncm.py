"""A module to hold some async specific functionality."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import override

if TYPE_CHECKING:
    from multiprocessing.connection import Connection


@dataclass
class AsyncConnection:
    """A wrapper around a multiprocessing connection to make it async."""

    connection: Connection
    """The underlying connection"""

    async def recv(self) -> Any:
        """Receive a message from the connection.

        Returns:
            The received message.
        """
        is_readable = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_reader(self.connection.fileno(), is_readable.set)

        if not self.connection.poll():
            await is_readable.wait()

        result = self.connection.recv()
        is_readable.clear()
        loop.remove_reader(self.connection.fileno())
        return result

    async def send(self, obj: Any) -> None:
        """Send a message to the connection.

        Args:
            obj: The object to send.
        """
        is_writable = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_writer(self.connection.fileno(), is_writable.set)

        await is_writable.wait()

        self.connection.send(obj)
        is_writable.clear()
        loop.remove_writer(self.connection.fileno())


class ContextEvent(asyncio.Event):
    """An event with added context to why it was triggered.

    Attributes:
        msg: The message that was set.
        exception: The exception that was set.
    """

    def __init__(self, **kwargs: Any) -> None:
        """See [asyncio.Event][] for more information."""
        super().__init__(**kwargs)
        self.msg: str | None = None
        self.exception: BaseException | None = None

    @override
    def set(
        self,
        msg: str | None = None,
        exception: BaseException | None = None,
    ) -> None:
        """Set the event and set the context.

        Args:
            msg: The message to set.
            exception: The exception to set.
        """
        self.msg = msg
        self.exception = exception
        super().set()

    @override
    def clear(self) -> None:
        """Clear the event and clear the context."""
        self.msg = None
        self.exception = None
        super().clear()

    @property
    def context(self) -> tuple[str | None, BaseException | None]:
        """Get the context information.

        Returns:
            A tuple of the message and exception.
        """
        return self.msg, self.exception
