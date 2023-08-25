from __future__ import annotations

import numpy as np
import pandas as pd

from amltk.metalearning import portfolio_selection


def test_max_portfolio():
    # The clearly best choice is config 1, followed by config 3 and lastly config 4
    portfolio = pd.DataFrame(
        [
            [2, 2, 0, 0],  # Best is config 1 (col1)
            [0, 0, 2, 0],  # Best is config 3 (col3)
            [0, 0, 1, 2],  # Best is config 4 (col4)
            [2, 1, 0, 0],  # Best is config 1 (col1)
        ],
        index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"],
        columns=["config_1", "config_2", "config_3", "config_4"],
        dtype=float,
    )

    # The scores here are normalized per dataset (row)
    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0.5, 1],
            [1, 0, 0],
        ],
        index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"],
        columns=["config_1", "config_3", "config_4"],
        dtype=float,
    )
    expected_trajectory = pd.Series({"config_1": 0.5, "config_3": 0.875, "config_4": 1})

    selected_portfolio, trajectory = portfolio_selection(
        items=portfolio.to_dict(orient="series"),
        k=3,
        # This is applied to each row => pd.DataFrame -> pd.Series
        row_reducer=np.max,
        # This is applied to the aggregation of each row => pd.Series -> float
        aggregator=np.mean,
        maximize=True,
        scaler="minmax",
        seed=1,
    )

    assert selected_portfolio.equals(expected)
    assert trajectory.equals(expected_trajectory)
