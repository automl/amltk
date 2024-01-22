from __future__ import annotations

from typing import Any


def threadpoolctl_heuristic(thing: Any | None) -> bool:
    """Heuristic to determine if we should automatically set threadpoolctl.

    Args:
        thing: The thing to check.

    Returns:
        Whether we should automatically set threadpoolctl.
    """
    if thing is None or not isinstance(thing, type):
        return False

    try:
        # NOTE: sklearn depends on threadpoolctl so it will be installed.
        from sklearn.base import BaseEstimator

        return issubclass(thing, BaseEstimator)
    except ImportError:
        return False
