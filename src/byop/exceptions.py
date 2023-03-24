"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

import traceback
from typing import Callable, Generic, ParamSpec, TypeVar

R = TypeVar("R")
E = TypeVar("E")
P = ParamSpec("P")


class exception_wrap(Generic[P, R]):  # noqa: N801
    """Wrap a function to add a traceback to any exception raised.

    Args:
        f: The function to wrap.
    """

    def __init__(self, f: Callable[P, R]) -> None:
        """Initialize the decorator."""
        self.f = f

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped function.

        Args:
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Raises:
            Exception: Any exception raised by the wrapped function will be
                re-raised with the traceback added to the error message.

        Returns:
            The return value of the wrapped function.
        """
        try:
            return self.f(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise attach_traceback(e) from e


def attach_traceback(exception: E) -> E:
    """Attach a traceback to the exception message.

    Args:
        exception: The exception to attach a traceback to.

    Returns:
        The exception with the traceback attached.
    """
    err_type: type[E] = type(exception)
    tb = traceback.format_exc()
    err_msg = str(exception)
    return err_type(f"{err_msg}\n{tb}")  # type: ignore
