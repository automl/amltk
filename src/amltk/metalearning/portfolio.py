"""A portfolio in meta-learning is to a set (ordered or not) of configurations
that maximize some notion of coverage across datasets or tasks.
The intuition here is that this also means that any new dataset is also covered!

Suppose we have the given performances of some configurations across some datasets.
```python exec="true" source="material-block" result="python" title="Initial Portfolio"
import pandas as pd

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])
print(portfolio)
```

If we could only choose `#!python k=3` of these configurations on some new given dataset, which ones would
you choose and in what priority?
Here is where we can apply [`portfolio_selection()`][amltk.metalearning.portfolio_selection]!

The idea is that we pick a subset of these algorithms that maximise some value of utility for
the portfolio. We do this by adding a single configuration from the entire set, 1-by-1 until
we reach `k`, beginning with the empty portfolio.

Let's see this in action!

```python exec="true" source="material-block" result="python" title="Portfolio Selection" hl_lines="12 13 14 15 16"
import pandas as pd
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax"
)

print(selected_portfolio)
print()
print(trajectory)
```

The trajectory tells us which configuration was added at each time stamp along with the utility
of the portfolio with that configuration added. However we havn't specified how _exactly_ we defined the
utility of a given portfolio. We could define our own function to do so:

```python exec="true" source="material-block" result="python" title="Portfolio Selection Custom" hl_lines="12 13 14 20"
import pandas as pd
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

def my_function(p: pd.DataFrame) -> float:
    # Take the maximum score for each dataset and then take the mean across them.
    return p.max(axis=1).mean()

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax",
    portfolio_value=my_function,
)

print(selected_portfolio)
print()
print(trajectory)
```

This notion of reducing across all configurations for a dataset and then aggregating these is common
enough that we can also directly just define these operations and we will perform the rest.

```python exec="true" source="material-block" result="python" title="Portfolio Selection With Reduction" hl_lines="17 18"
import pandas as pd
import numpy as np
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax",
    row_reducer=np.max,  # This is actually the default
    aggregator=np.mean,  # This is actually the default
)

print(selected_portfolio)
print()
print(trajectory)
```
"""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Literal, TypeVar

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
            best_keys[0] if len(best_keys) == 1 else rng.choice(best_keys)  # type: ignore
        )

        # We found something better, add it in
        added_items.append(best_key)
        values.append(best_possible)

        if not with_replacement:
            del items[best_key]

    # Rename the columns of the portfolio to be the keys
    return all_portfolio[added_items], pd.Series(values, index=added_items)
