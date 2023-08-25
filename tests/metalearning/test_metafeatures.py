from __future__ import annotations

import numpy as np
import pandas as pd
from more_itertools import take
from pytest_cases import case, parametrize, parametrize_with_cases
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

from amltk.metalearning import compute_metafeatures


def categorize(x: pd.DataFrame, n_cols: int, bins: int = 5) -> pd.DataFrame:
    """Categorizes a series."""
    for col in take(n_cols, x.columns):
        x[col] = pd.cut(x[col], bins=bins)

    return x


def inject_missing(x: pd.DataFrame, percentage_missing: float) -> pd.DataFrame:
    # Inject missing values
    if percentage_missing > 0:
        rng = np.random.default_rng(1)
        rows, cols = np.where(rng.random(x.shape) < 0.1)
        x.iloc[rows, cols] = np.nan

    return x


@case
@parametrize(n_categories=[0, 2, 5])
@parametrize(percentage_missing=[0, 0.1, 0.5])
def case_binary_classification(
    n_categories: int,
    percentage_missing: float,
) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(  # type: ignore
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=0,
    )
    x = pd.DataFrame(x)
    y = pd.Series(y).astype("category")

    x = categorize(x, n_categories)
    x = inject_missing(x, percentage_missing)

    return x, y


@case
@parametrize(n_categories=[0, 2, 5])
@parametrize(n_classes=[3, 5])
@parametrize(percentage_missing=[0, 0.1, 0.5])
def case_multiclass_classification(
    n_classes: int,
    n_categories: int,
    percentage_missing: float,
) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(  # type: ignore
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=0,
        n_classes=n_classes,
    )
    x = pd.DataFrame(x)
    y = pd.Series(y).astype("category")

    x = categorize(x, n_categories)
    x = inject_missing(x, percentage_missing)

    return x, y


@case
@parametrize(n_categories=[0, 2, 5])
@parametrize(percentage_missing=[0, 0.1, 0.5])
def case_multilabel_binary_classification(
    n_categories: int,
    percentage_missing: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = make_multilabel_classification(  # type: ignore
        n_samples=1000,
        n_features=10,
        n_classes=5,
        n_labels=2,
        random_state=0,
    )
    x, y = pd.DataFrame(x), pd.DataFrame(y).astype("category")

    x = categorize(x, n_categories)
    x = inject_missing(x, percentage_missing)

    return x, y


@case
@parametrize(n_categories=[0, 2, 5])
@parametrize(percentage_missing=[0, 0.1, 0.5])
def case_single_target_regression(
    n_categories: int,
    percentage_missing: float,
) -> tuple[pd.DataFrame, pd.Series]:
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=0,
    )

    x, y = pd.DataFrame(x), pd.Series(y)

    x = categorize(x, n_categories)
    x = inject_missing(x, percentage_missing)

    return x, y


@case
@parametrize(n_categories=[0, 2, 5])
@parametrize(n_targets=[2, 5])
@parametrize(percentage_missing=[0, 0.1, 0.5])
def case_multitarget_regression(
    n_categories: int,
    n_targets: int,
    percentage_missing: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = make_regression(  # type: ignore
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=0,
        n_targets=n_targets,
    )
    x, y = pd.DataFrame(x), pd.DataFrame(y)

    x = categorize(x, n_categories)
    x = inject_missing(x, percentage_missing)

    return x, y


@parametrize_with_cases(argnames=["x", "y"], cases=".")
def test_metafeature_computation(x: pd.DataFrame, y: pd.Series | pd.DataFrame) -> None:
    mfs = compute_metafeatures(x, y)

    is_classification = (
        all(y.dtypes == "category")
        if isinstance(y, pd.DataFrame)
        else y.dtype == "category"
    )
    any_missing = x.isna().any().any()
    any_categorical = any(x.dtypes == "category")

    assert not mfs.isna().any()

    if is_classification:
        assert mfs["number_of_classes"] > 0
        assert mfs["class_imbalance"] > 0
        assert mfs["majority_class_imbalance"] > 0
        assert mfs["minority_class_imbalance"] > 0

    if any_categorical:
        assert mfs["number_of_categorical_features"] > 0
        assert mfs["ratio_categorical_features"] > 0
        assert mfs["ratio_numerical_features"] < 1

        assert mfs["mean_categorical_imbalance"] > 0
        assert mfs["std_categorical_imbalance"] > 0
    else:
        assert mfs["ratio_numerical_features"] == 1.0
        assert mfs["ratio_categorical_features"] == 0.0

    if any_missing:
        assert mfs["percentage_of_features_with_missing_values"] > 0
        assert mfs["percentage_of_instances_with_missing_values"] > 0
        assert mfs["percentage_missing_values"] > 0

        assert mfs["percentage_of_numeric_columns_with_missing_values"] > 0
        assert mfs["percentage_of_numeric_values_with_missing_values"] > 0

    if any_categorical and any_missing:
        assert mfs["percentage_of_categorical_columns_with_missing_values"] > 0
        assert mfs["percentage_of_categorical_values_with_missing_values"] > 0
