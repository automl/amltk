"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar
from typing_extensions import ParamSpec

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

    def __repr__(self) -> str:
        """Get the string representation of the decorator.

        Returns:
            The string representation of the decorator.
        """
        return self.f.__repr__()


class _KeyErrorMessage(str):
    """This is a throw-away class to get around an issue with KeyErrors.

    Namely, any strings in a KeyError end up having all special characters
    escaped.

    See Also:
        * https://bugs.python.org/issue2651#msg65587
    """

    def __repr__(self) -> str:
        return str(self)


def attach_traceback(exception: E) -> E:
    """Attach a traceback to the exception message.

    Args:
        exception: The exception to attach a traceback to.

    Returns:
        The exception with the traceback attached.
    """
    err_type: type[E] = type(exception)
    tb = traceback.format_exc()
    err_msg = "\n".join([str(exception), str(tb)])

    # See class doc of _KeyErrorMessage for why this is needed.
    if err_type is KeyError:
        err_msg = _KeyErrorMessage(err_msg)

    try:
        return err_type(err_msg)  # type: ignore
    except Exception:  # noqa: BLE001
        return exception


def safe_map(
    f: Callable[..., R],
    args: Iterable[Any],
    *,
    attached_tb: bool = False,
) -> Iterator[R | Exception]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.
        attached_tb: Whether to attach a traceback to the exception.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(arg)
        except Exception as e:  # noqa: BLE001
            yield attach_traceback(e) if attached_tb else e


def safe_starmap(
    f: Callable[..., R],
    args: Iterable[Iterable[Any]],
    *,
    attached_tb: bool = False,
) -> Iterator[R | Exception]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.
        attached_tb: Whether to attach a traceback to the exception.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(*arg)
        except Exception as e:  # noqa: BLE001
            yield attach_traceback(e) if attached_tb else e
