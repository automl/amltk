from __future__ import annotations

from typing import Any


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
