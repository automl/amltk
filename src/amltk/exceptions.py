"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import ParamSpec, override

if TYPE_CHECKING:
    from amltk.pipeline.node import Node

R = TypeVar("R")
E = TypeVar("E")
P = ParamSpec("P")


def safe_map(
    f: Callable[..., R],
    args: Iterable[Any],
) -> Iterator[R | tuple[Exception, str]]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(arg)
        except Exception as e:  # noqa: BLE001
            yield e, traceback.format_exc()


def safe_starmap(
    f: Callable[..., R],
    args: Iterable[Iterable[Any]],
) -> Iterator[R | tuple[Exception, str]]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(*arg)
        except Exception as e:  # noqa: BLE001
            yield e, traceback.format_exc()


class IntegrationNotFoundError(Exception):
    """An exception raised when no integration is found."""

    def __init__(self, name: str) -> None:
        """Initialize the exception.

        Args:
            name: The name of the integration that was not found.
        """
        super().__init__(f"No integration found for {name}.")


class SchedulerNotRunningError(RuntimeError):
    """The scheduler is not running."""


class EventNotKnownError(ValueError):
    """The event is not a known one."""


class NoChoiceMadeError(ValueError):
    """No choice was made."""


class NodeNotFoundError(ValueError):
    """The node was not found."""


class RequestNotMetError(ValueError):
    """Raised when a request is not met."""


class DuplicateNamesError(ValueError):
    """Raised when duplicate names are found."""

    def __init__(self, node: Node) -> None:
        """Initialize the exception.

        Args:
            node: The node that has children with duplicate names.
        """
        super().__init__(node)
        self.node = node

    @override
    def __str__(self) -> str:
        return (
            f"Duplicate names found in {self.node.name} and can't be handled."
            f"\nnodes: {[n.name for n in self.node.nodes]}."
        )
