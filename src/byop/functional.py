"""A collection of functional programming tools.

This module contains a collection of tools for functional programming.
"""
from __future__ import annotations

from functools import partial, reduce
from itertools import count
from types import EllipsisType
from typing import Any, Callable, Iterator, Reversible, Sequence, TypeVar

T = TypeVar("T")


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


def reposition(
    items: Sequence[T],
    order: Reversible[T | EllipsisType],
) -> list[T]:
    """Reposition values of a sequence.

    This function assumes only one ellipsis is present in the order.
    There is no garuntee of functionallity if this is not the case.
    It's also not very efficient, but it's good enough for now.

    ```python
    xs = [1, 2, 3, 4, 5]

    reposition(xs, [..., 1])        # [2, 3, 4, 5, 1]
    reposition(xs, [5, ...])        # [5, 1, 2, 3, 4]
    reposition(xs, [5, ..., 1])     # [5, 2, 3, 4, 1]
    ```

    Args:
        order: The order to reposition the items in.
            The Ellipses can be used to indicate where the remaining
            items should be placed.
        items: The items to reposition.

    Returns:
        The repositioned items.
    """
    front_values = []
    indicies_taken: set[int] = set()

    for item in order:
        if item is Ellipsis:
            break
        index = items.index(item)
        indicies_taken.add(index)
        front_values.append(items[index])

    back_values = []
    for item in reversed(order):
        if item is Ellipsis:
            break
        index = items.index(item)
        indicies_taken.add(index)
        back_values.append(items[index])

    middle_items = [v for i, v in enumerate(items) if i not in indicies_taken]

    return front_values + middle_items + back_values[::-1]


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


def funcname(func: Callable) -> str:
    """Get the name of a function.

    Args:
        func: The function to get the name of.

    Returns:
        The name of the function.
    """
    if isinstance(func, partial):
        return func.func.__name__
    return func.__qualname__
