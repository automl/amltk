"""Data utilities for scikit-learn."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING

from more_itertools import last
from sklearn.model_selection import train_test_split

from amltk.randomness import as_int

if TYPE_CHECKING:
    from amltk.types import Seed


def split_data(
    *items: Sequence,
    splits: dict[str, float],
    seed: Seed | None = None,
    shuffle: bool = True,
    stratify: Sequence | None = None,
) -> dict[str, tuple[Sequence, ...]]:
    """Split a set of items into multiple splits.

    ```python
    from amltk.sklearn.data import split

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    splits = split_data(x, y, splits={"train": 0.6, "val": 0.2, "test": 0.2})

    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]
    ```

    Args:
        items: The items to split. Must be indexible, like a list, np.ndarray,
            pandas dataframe/series or a tuple, etc...
        splits: A dictionary of split names and their percentage of the data.
            The percentages must sum to 1.
        seed: The seed to use for the random state.
        shuffle: Whether to shuffle the data before splitting. Passed forward
            to [sklearn.model_selection.train_test_split][].
        stratify: The stratification to use for the split. This will be passed
            forward to [sklearn.model_selection.train_test_split][]. We account
            for using the stratification for all splits, ensuring we split of
            the stratification values themselves.

    Returns:
        A dictionary of split names and their split items.
    """
    if not all(0 < s < 1 for s in splits.values()):
        raise ValueError(f"Splits ({splits=}) must be between 0 and 1")

    if sum(splits.values()) != 1:
        raise ValueError(f"Splits ({splits=}) must sum to 1")

    if len(splits) < 2:  # noqa: PLR2004
        raise ValueError(f"Splits ({splits=}) must have at least 2 splits")

    rng = as_int(seed) if seed is not None else None

    # Store the results of each split, indexed by the split number
    split_results: dict[str, list[Sequence]] = {}
    remaining: list[Sequence] = list(items)

    remaining_percentage = 1.0

    # Enumerate up to the last split
    for name, split_percentage in list(splits.items())[0:-1]:
        # If we stratify, make sure to also include it in the splitting so
        # further splits can be stratified correctly.
        to_split = remaining if stratify is None else [*remaining, stratify]

        # Calculate the percentage of the remaining data to split
        percentage = split_percentage / remaining_percentage

        splitted = train_test_split(
            *to_split,
            train_size=percentage,
            random_state=rng,
            shuffle=shuffle,
            stratify=stratify,
        )

        # Update the remaining percentage
        remaining_percentage -= split_percentage

        # Splitted returns pairs of (train, test) for each item in items
        # so we need to split them up
        lefts = splitted[::2]
        rights = splitted[1::2]

        # If we had stratify, we need to remove the last item from splits
        # as it was the stratified array, setting the stratification for
        # the next split
        if stratify is not None:
            stratify = rights[-1]  # type: ignore

            lefts = lefts[:-1]
            rights = rights[:-1]

        # Lastly, we insert the lefts into the split_results
        # and set the remaining to the rights
        split_results[name] = lefts  # type: ignore
        remaining = rights  # type: ignore

    # Since we enumerated up to the last split, we need to add the last
    # split manually
    last_name = last(splits.keys())
    split_results[last_name] = remaining

    return {name: tuple(split) for name, split in split_results.items()}


def train_val_test_split(
    *items: Sequence,
    splits: tuple[float, float, float],
    seed: Seed | None = None,
    shuffle: bool = True,
    stratify: Sequence | None = None,
) -> tuple[Sequence, ...]:
    """Split a set of items into train, val and test splits.

    ```python
    from amltk.sklearn.data import train_val_test_split

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(
        x, y, splits=(0.6, 0.2, 0.2),
    )
    ```

    Args:
        items: The items to split. Must be indexible, like a list, np.ndarray,
            pandas dataframe/series or a tuple, etc...
        splits: A tuple of the percentage of the data to use for the train,
            val and test splits. The percentages must sum to 1.
        seed: The seed to use for the random state.
        shuffle: Whether to shuffle the data before splitting. Passed forward
            to [sklearn.model_selection.train_test_split][].
        stratify: The stratification to use for the split. This will be passed
            forward to [sklearn.model_selection.train_test_split][]. We account
            for using the stratification for all splits, ensuring we split of
            the stratification values themselves.

    Returns:
        A tuple containing the train, val and test splits.
    """
    results = split_data(
        *items,
        splits={"train": splits[0], "val": splits[1], "test": splits[2]},
        seed=seed,
        shuffle=shuffle,
        stratify=stratify,
    )
    return tuple(chain(results["train"], results["val"], results["test"]))
