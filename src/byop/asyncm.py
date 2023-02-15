"""A module to hold some async specific functionality."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any


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
