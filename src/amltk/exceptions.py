"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Iterable, Iterator, TypeVar
from typing_extensions import ParamSpec

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