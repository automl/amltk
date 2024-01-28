from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import datetime
from typing import Any

import pandas as pd


def threadpoolctl_heuristic(item_contained_in_node: Any | None) -> bool:
    """Heuristic to determine if we should automatically set threadpoolctl.

    This is done by detecting if it's a scikit-learn `BaseEstimator` but this may
    be extended in the future.

    !!! tip

        The reason to have this heuristic is that when running scikit-learn, or any
        multithreaded model, in parallel, they will over subscribe to threads. This
        causes a significant performance hit as most of the time is spent switching
        thread contexts instead of work. This can be particularly bad for HPO where
        we are evaluating multiple models in parallel on the same system.

        The recommened thread count is 1 per core with no additional information to
        act upon.

    !!! todo

        This is potentially not an issue if running on multiple nodes of some cluster,
        as they do not share logical cores and hence do not clash.

    Args:
        item_contained_in_node: The item with which to base the heuristic on.

    Returns:
        Whether we should automatically set threadpoolctl.
    """
    if item_contained_in_node is None or not isinstance(item_contained_in_node, type):
        return False

    try:
        # NOTE: sklearn depends on threadpoolctl so it will be installed.
        from sklearn.base import BaseEstimator

        return issubclass(item_contained_in_node, BaseEstimator)
    except ImportError:
        return False


def parse_timestamp_object(timestamp: Any) -> datetime:
    """Parse a timestamp object, erring if it can't be parsed.

    Args:
        timestamp: The timestamp to parse.

    Returns:
        The parsed timestamp or `None` if it could not be parsed.
    """
    # Make sure we correctly set it's generated at if
    # we can
    match timestamp:
        case datetime():
            return timestamp
        case pd.Timestamp():
            return timestamp.to_pydatetime()
        case float() | int():
            return datetime.fromtimestamp(timestamp)
        case str():
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse `str` type timestamp for '{timestamp}'."
                    " \nPlease provide a valid isoformat timestamp, e.g."
                    "'2021-01-01T00:00:00.000000'.",
                ) from e
        case _:
            raise TypeError(f"Could not parse {timestamp=} of type {type(timestamp)}.")


@contextmanager
def ignore_warnings() -> Iterator[None]:
    """Ignore warnings for the duration of the context manager."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@contextmanager
def mutli_context(*managers: AbstractContextManager) -> Iterator:
    """Run multiple context managers at once."""
    with ExitStack() as stack:
        yield [stack.enter_context(m) for m in managers]
