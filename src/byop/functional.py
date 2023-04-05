"""A collection of functional programming tools.

This module contains a collection of tools for functional programming.
"""
from __future__ import annotations

from functools import partial, reduce
from itertools import count
from typing import Any, Callable, Hashable, Iterator, Sequence, TypeVar

from byop.exceptions import exception_wrap

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
VK = TypeVar("VK", bound=Hashable)

from byop.exceptions import exception_wrap


def reverse_enumerate(
    seq: Sequence[T],
    start: int | None = None,
) -> Iterator[tuple[int, T]]:
    """Reverse enumerate.

    This function is similar to enumerate, but it iterates over the
    sequence in reverse.

    ```python
    xs = ["a", "b", "c"]
    for i, x in reverse_enumerate(xs):
        print(i, x)
    # 2 c
    # 1 b
    # 0 a

    Args:
        seq: The sequence to iterate over.
        start: The starting index.

    Returns:
        An iterator over the sequence.
    """
    if start is None:
        start = len(seq) - 1
    yield from zip(count(start, -1), reversed(seq))


def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """Recursive getattr.

    This function is similar to getattr, but it allows you to get
    attributes using '.' notation.

    ```python
    class A:
        x = 1

    class B:
        a = A()

    b = B()
    rgetattr(b, "a.x")  # 1
    ```

    https://stackoverflow.com/a/31174427/5332072

    Args:
        obj: The object to get the attribute from.
        attr: The attribute to get.
        *args: The default value to return if the attribute is not found.

    Returns:
        The attribute.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj, *attr.split(".")])


def funcname(func: Callable, default: str | None = None) -> str:
    """Get the name of a function.

    Args:
        func: The function to get the name of.
        default: The default value to return if the name cannot be
            determined automatically.

    Returns:
        The name of the function.
    """
    if isinstance(func, exception_wrap):
        func = func.f

    if isinstance(func, partial):
        return func.func.__name__
    if hasattr(func, "__qualname__"):
        return func.__qualname__
    if hasattr(func, "__class__"):
        return func.__class__.__name__
    if default is not None:
        return default
    return str(func)


def callstring(f: Callable, *args: Any, **kwargs: Any) -> str:
    """Get a string representation of a function call.

    Args:
        f: The function to get the string representation of.
        *args: The positional arguments.
        **kwargs: The keyword arguments.

    Returns:
        The string representation of the function call.
    """
    # Iterate over all args, convert them to str, and join them
    args_str = ""
    if any(args):
        args_str += ", ".join(map(str, args))
    if any(kwargs):
        args_str += ", ".join(f"{k}={v}" for k, v in kwargs.items())

    return f"{funcname(f)}({args_str})"
