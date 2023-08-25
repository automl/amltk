"""Portfolio selection for meta-learning."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Hashable, Literal, TypeVar

import numpy as np
import pandas as pd

from amltk.randomness import as_rng
from amltk.types import Seed, safe_isinstance

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

K = TypeVar("K", bound=Hashable)


def portfolio_selection(
    items: dict[K, pd.Series] | pd.DataFrame,
    k: int,
    *,
    row_reducer: Callable[[pd.Series], float] = np.max,
    aggregator: Callable[[pd.Series], float] = np.mean,
    portfolio_value: Callable[[pd.DataFrame], float] | None = None,
    maximize: bool = True,
    scaler: TransformerMixin | Literal["minmax"] | None = "minmax",
    with_replacement: bool = False,
    stop_if_worse: bool = False,
    seed: Seed | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Selects a portfolio of `k` items from `items`.

    A portfolio is a subset of the items, and is selected by maximizing the
    `portfolio_value` function in a greedy selection approach.

    At each iteration `0 <= i < k`, the `portfolio_value` function is calculated
    for the portfolio obtained by adding the `i`th item to the portfolio. The item
    that maximizes the `portfolio_value` function is then added to the portfolio for
    the next iteration.

    The `portfolio_function` can often be define by a row wise reduction
    (`row_reducer=`) followed by some aggregation over these reductions (`aggregator=`).
    You can also supply your own value function if desired (`portfolio_value=`).

    !!! example "A Single Iteration"

        This uses the `row_reducer=np.max` and `aggregator=np.mean` to calculate the
        value of a portfolio.

        In this case, we have 4 datasets and our current portfolio
        consists of `config_1` and `config_2`. We are going to calculate the value of
        adding `config_try` to the current best portfolio.

        ```python
                    | config_1 | config_2 | config_try
        dataset_1   |    1     |    0     |    0
        dataset_2   |    0     |   0.5    |    1
        dataset_3   |    0     |   0.5    |   0.5
        dataset_4   |    1     |    1     |    0
        ```

        Apply `row_reducer` to each row, in this case `np.max`

        ```python
                    |   max
        dataset_1   |    1
        dataset_2   |    1
        dataset_3   |   0.5
        dataset_4   |    1
        ```

        Apply `aggregator` to the reduced rows, in this case `np.mean`

        ```python
        portfolio_value = np.mean([1, 1, 0.5, 1]) # 0.875
        ```

    Args:
        items: A dictionary of items to select from.
        k: The number of items to select.
        row_reducer: A function to aggregate the rows of the portfolio.
            This is applied to a potential portfolio, for example to calculate
            the max score of all configs, for a given dataset (row).
        aggregator: A function to take all the single values reduced by `row_reducer`,
            and aggregate them into a final value for the portfolio.
        portfolio_value: A custom function to calculate the value of a portfolio.
            This will take precedence over `row_reducer` and `aggregator`.
        maximize: Whether to maximize or minimize the portfolio value.
        scaler: A scaler to use to scale the portfolio values. Is applied across
            the rows.
        with_replacement: Whether to select items with replacement.
        stop_if_worse: Whether to stop if the portfolio value is worse than the
            current best.
        seed: The seed to use for breaking ties.

    Returns:
        The final portfolio
        The trajectory, where the entry is the value once added to the portfolio.
    """
    if not (1 <= k < len(items)):
        raise ValueError(f"k must be in [1, {len(items)=})")

    all_portfolio = pd.DataFrame(items)

    # Normalize if needed
    if scaler is None:
        pass
    elif scaler == "minmax":
        min_maxs = all_portfolio.agg(["min", "max"], axis=1)

        mins = min_maxs["min"]
        maxs = min_maxs["max"]
        normalizer = maxs - mins

        # If everything is equal, we need to make sure the normalizing
        # doesn't do anything
        normalizer[normalizer == 0] = 1
        mins[normalizer == 0] = 0

        norm = lambda col: (col - mins) / normalizer
        all_portfolio: pd.DataFrame = all_portfolio.apply(norm, axis=0)  # type: ignore
    elif safe_isinstance(scaler, "TransformerMixin"):
        assert not isinstance(scaler, str)
        all_portfolio = scaler.fit_transform(all_portfolio.T).T
    else:
        raise ValueError(f"Invalid scaler: {scaler}")

    # Set up the portfolio value function
    if portfolio_value is None:
        portfolio_value = lambda _portfolio: float(
            aggregator(_portfolio.apply(row_reducer, axis=1)),
        )

    # Make a copy as we will del from it
    items = dict(items)
    rng = as_rng(seed)
    best = max if maximize else min

    # Running counters during the algorithm loop
    added_items: list[K] = []
    values: list[float] = []
    current_best: float = -np.inf if maximize else np.inf

    for _ in range(k):
        possible_portfolios = [(k, all_portfolio[[*added_items, k]]) for k in items]
        values_possible = {
            k: portfolio_value(possible_portfolio)
            for k, possible_portfolio in possible_portfolios
        }

        # This is the highest value we can get from a portfolio of the current size
        best_possible = best(values_possible.values())

        # If the best possible value of what we can do does not improve over the current
        #    best portfolio, stop (if enabled)
        if stop_if_worse and current_best == best(best_possible, current_best):
            break

        current_best = best_possible

        # Possible get multiple best choices, we choose one at random if so
        best_keys = [k for k, v in values_possible.items() if v == best_possible]
        best_key = (
            best_keys[0]
            if len(best_keys) == 1
            else rng.choice(best_keys)  # type: ignore
        )

        # We found something better, add it in
        added_items.append(best_key)
        values.append(best_possible)

        if not with_replacement:
            del items[best_key]

    # Rename the columns of the portfolio to be the keys
    return all_portfolio[added_items], pd.Series(values, index=added_items)
