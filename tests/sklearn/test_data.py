from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from more_itertools import exactly_n
from pytest_cases import parametrize

from amltk.sklearn.data import split_data


@parametrize(
    "items",
    [
        (list(range(100)), list(range(100))),
        (np.arange(100), np.arange(100)),
        (pd.Series(range(100)), pd.Series(range(100))),
        (pd.DataFrame(range(100)), pd.DataFrame(range(100))),
    ],
)
@parametrize("shuffle", [True, False])
@parametrize(
    "splits",
    [
        {"train": 0.6, "val": 0.2, "test": 0.2},
        {"train": 0.8, "test": 0.2},
    ],
)
def test_split_data(
    items: tuple[Sequence, ...],
    shuffle: bool,
    splits: dict[str, float],
) -> None:
    _splits = split_data(*items, splits=splits, seed=42, shuffle=shuffle)
    assert len(_splits) == len(splits)
    assert all(len(split) == len(items) for split in _splits.values())
    assert all(len(split[0]) == len(split[1]) for split in _splits.values())

    for key, split_items in _splits.items():
        percentage = splits[key]
        for item in split_items:
            assert len(item) == int(percentage * len(items[0]))


def test_split_data_stratified() -> None:
    # 20 (1s) and 10 (0s)
    X = list(range(30))
    y = [0] * 10 + [1] * 20

    split_percentages = {"train": 0.6, "val": 0.2, "test": 0.2}
    splits = split_data(
        X,
        y,
        splits=split_percentages,
        stratify=y,
    )

    is_one = lambda x: x == 1
    is_zero = lambda x: x == 0

    _, train_y = splits["train"]
    assert exactly_n(train_y, 12, is_one)
    assert exactly_n(train_y, 6, is_zero)

    _, val_y = splits["val"]
    assert exactly_n(val_y, 4, is_one)
    assert exactly_n(val_y, 2, is_zero)

    _, test_y = splits["test"]
    assert exactly_n(test_y, 4, is_one)
    assert exactly_n(test_y, 2, is_zero)
