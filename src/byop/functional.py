"""A collection of functional programming tools.

This module contains a collection of tools for functional programming.
"""
from __future__ import annotations

from itertools import count
from types import EllipsisType
from typing import Iterator, Reversible, Sequence, TypeVar

T = TypeVar("T")


def reverse_enumerate(
    seq: Sequence[T],
    start: int | None = None,
) -> Iterator[tuple[int, T]]:
    """Reverse enumerate."""
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
