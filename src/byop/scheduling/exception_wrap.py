"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Callable, Generic, ParamSpec, TypeVar

R = TypeVar("R")
P = ParamSpec("P")


@dataclass
class exception_wrap(Generic[P, R]):  # noqa: N801
    """Wrap a function to add a traceback to any exception raised.

    Args:
        f: The function to wrap.
    """

    f: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped function.

        Args:
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Raises:
            Any exception raised by the wrapped function will be re-raised
            with the traceback added to the error message.

        Returns:
            The return value of the wrapped function.
        """
        try:
            return self.f(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            err_type = type(e)
            tb = traceback.format_exc()
            err_msg = str(e)
            raise err_type(f"{err_msg}\n{tb}") from e
