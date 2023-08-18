"""Metafeatures for use in metalearning."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generic, Tuple, TypeVar
from typing_extensions import TypeAlias, override

import numpy as np
import pandas as pd

CAMEL_CASE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")

if TYPE_CHECKING:
    DSdict: TypeAlias = Dict[type["DatasetStatistic"], Any]

S = TypeVar("S")
M = TypeVar("M", bound=float)
T = TypeVar("T")

KURTOSIS_OF_NORMAL_DISTRIBUTION = 0.0
SKEWNESS_OF_NORMAL_DISTRIBUTION = 0.0


def imbalance_ratios(col: pd.Series) -> tuple[np.ndarray, float]:
    """Compute the imbalance ratio of a categorical column.

    This is done by computing the distance of each item's ratio to what
    a perfectly balanced ratio would be. We then sum up the distances,
    dividing by the worst case to normalize between 0 and 1.

    Args:
        col: A column of values.

    Returns:
        A tuple of the imbalance ratios, sorted from lowest (0) to highest (1)
        and the expected ratio if perfectly balanced.
    """
    uniq_items = np.unique(col, return_counts=True)[1]
    if len(uniq_items) == 1:
        return np.asarray([1.0]), 1.0

    n_uniq = len(uniq_items)
    n_items = len(col)

    ratios = uniq_items / n_items

    # A balanced ratio is one where all items are equally distributed
    balanced_ratio = 1 / n_uniq
    ratios.sort()
    return ratios, balanced_ratio


def column_imbalance(ratios: np.ndarray, balanced_ratio: float) -> float:
    """Compute the imbalance of a column.

    This is done by computing the distance of each item's ratio to what
    a perfectly balanced ratio would be. We then sum up the distances,
    dividing by the worst case to normalize between 0 and 1. 0 indicates
    a perfectly balanced column, 1 indicates a column where all items
    are of the same type.

    Args:
        ratios: The ratios of each item in the column.
        balanced_ratio: The ratio of a column if perfectly balanced.

    Returns:
        The imbalance of the column.
    """
    item_ratios_distance_from_balanced_ratio = np.abs(ratios - balanced_ratio)

    # The most imbalanced dataset would be one where we somehow have 0
    # items of each type **except** 1 type, which has all the instances.

    # In the case of a symbol group with 0 instance, their distance to the balanced
    # ratio is just the balanced ratio itself.
    zero_instance_ratio_distance = balanced_ratio
    dominant_ratio_distance = np.abs(1 - balanced_ratio)
    n_items = len(ratios)

    worst = (n_items - 1) * zero_instance_ratio_distance + dominant_ratio_distance
    normalizer = 1 / worst

    return float(normalizer * np.sum(item_ratios_distance_from_balanced_ratio))


class DatasetStatistic(ABC, Generic[S]):
    """Base class for a dataset statistic.

    A dataset statistic is a function that takes a dataset and returns some
    value(s) that describe the dataset.

    If looking to create meta-features, see the `MetaFeature` class which
    restricts the statistic to be a single number.
    """

    dependencies: ClassVar[tuple[type[DatasetStatistic], ...]] = ()

    @classmethod
    def description(cls) -> str:
        """Return the description of this statistic."""
        return cls.__doc__ or ""

    @classmethod
    def name(cls) -> str:
        """Return the name of this statistic."""
        return CAMEL_CASE_PATTERN.sub("_", cls.__name__).lower()

    @classmethod
    @abstractmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> S:
        """Compute the value of this statistic.

        Args:
            x: The features of the dataset.
            y: The labels of the dataset.
            dependancy_values: A dictionary of dependency values.

        Returns:
            The value of this statistic.
        """

    @classmethod
    def retrieve(cls, dependancy_values: DSdict) -> S:
        """Retrieve the value of this statistic from the dependency values.

        Args:
            dependancy_values: A dictionary of dependency values.

        Returns:
            The value of this statistic.
        """
        return dependancy_values[cls]


class MetaFeature(DatasetStatistic[M]):
    """Used to indicate a metafeature to include.

    This differs from DatasetStatistic in that it must return a single value.
    """


class NAValues(DatasetStatistic[pd.DataFrame]):
    """Mask of missing values in the dataset."""

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> pd.DataFrame:
        return x.isna()


class ClassImbalanceRatios(DatasetStatistic[Tuple[np.ndarray, float]]):
    """Imbalance ratios of each class in the dataset.

    Will return the ratios of each class, the ratio expected if perfectly balanced,
    """

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> tuple[np.ndarray, float]:
        return imbalance_ratios(y)


class CategoricalImbalanceRatios(DatasetStatistic[Dict[str, Tuple[np.ndarray, float]]]):
    """Imbalance ratios of each class in the dataset.

    Will return the ratios of each class, the ratio expected if perfectly balanced,
    """

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> dict[str, tuple[np.ndarray, float]]:
        categorical_columns = x.select_dtypes(exclude=np.number).columns
        return {c: imbalance_ratios(x[c]) for c in categorical_columns}


class CategoricalColumns(DatasetStatistic[pd.DataFrame]):
    """The categorical columns in the dataset."""

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> pd.DataFrame:
        return x.select_dtypes(exclude=np.number)


class NumericalColumns(DatasetStatistic[pd.DataFrame]):
    """The numerical columns in the dataset."""

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> pd.DataFrame:
        return x.select_dtypes(include=np.number)


class InstanceCount(MetaFeature[int]):
    """Number of instances in the dataset."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> int:
        return x.shape[0]


class LogInstanceCount(MetaFeature[float]):
    """Logarithm of the number of instances in the dataset."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        return np.log(x.shape[0])


class NumberOfClasses(MetaFeature[int]):
    """Number of classes in the dataset."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> int:
        if len(y.shape) == 1:
            return len(np.unique(y))

        if len(y.shape) == 2:  # noqa: PLR2004
            return y.shape[1]

        raise ValueError("y must be 1D or 2D")


class NumberOfFeatures(MetaFeature[int]):
    """Number of features in the dataset."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> int:
        return x.shape[1]


class LogNumberOfFeatures(MetaFeature[float]):
    """Logarithm of the number of features in the dataset."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        return float(np.log(x.shape[1]))


class PercentageMissingValues(MetaFeature[float]):
    """Percentage of missing values in the dataset."""

    dependencies = (NAValues,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        n_values = x.shape[0] * x.shape[1]
        na_values = NAValues.retrieve(dependancy_values)
        n_missing = na_values.sum().sum()
        return float(n_missing / n_values)


class PercentageOfInstancesWithMissingValues(MetaFeature[float]):
    """Percentage of instances with missing values."""

    dependencies = (NAValues,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        n_instances = x.shape[0]
        na_values = NAValues.retrieve(dependancy_values)
        n_instances_with_missing_values = na_values.any(axis=1).sum()
        return float(n_instances_with_missing_values / n_instances)


class PercentageOfFeaturesWithMissingValues(MetaFeature[float]):
    """Percentage of features with missing values."""

    dependencies = (NAValues,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        n_features = x.shape[1]
        na_values = NAValues.retrieve(dependancy_values)
        n_features_with_missing_values = na_values.any(axis=0).sum()
        return float(n_features_with_missing_values / n_features)


class PercentageOfCategoricalColumnsWithMissingValues(MetaFeature[float]):
    """Percentage of categorical columns with missing values."""

    dependencies = (CategoricalColumns, NAValues)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        categorical_columns = CategoricalColumns.retrieve(dependancy_values)
        na_values = NAValues.retrieve(dependancy_values)
        n_categorical_features = categorical_columns.shape[1]
        n_categorical_features_with_missing_values = (
            na_values[categorical_columns.columns].any(axis=0).sum()
        )
        return float(
            n_categorical_features_with_missing_values / n_categorical_features,
        )


class PercentageOfCategoricalValuesWithMissingValues(MetaFeature[float]):
    """Percentage of categorical values with missing values."""

    dependencies = (CategoricalColumns, NAValues)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        categorical_columns = CategoricalColumns.retrieve(dependancy_values)
        if categorical_columns.empty:
            return 0.0

        na_values = NAValues.retrieve(dependancy_values)
        n_categorical_values = (
            categorical_columns.shape[0] * categorical_columns.shape[1]
        )
        n_categorical_values_with_missing_values = (
            na_values[categorical_columns.columns].sum().sum()
        )
        return float(n_categorical_values_with_missing_values / n_categorical_values)


class PercentageOfNumericColumnsWithMissingValues(MetaFeature[float]):
    """Percentage of numeric columns with missing values."""

    dependencies = (NumericalColumns, NAValues)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        numerical_columns = NumericalColumns.retrieve(dependancy_values)
        if numerical_columns.empty:
            return 0.0

        na_values = NAValues.retrieve(dependancy_values)
        n_numerical_features = numerical_columns.shape[1]
        n_numerical_features_with_missing_values = (
            na_values[numerical_columns.columns].any(axis=0).sum()
        )
        return float(n_numerical_features_with_missing_values / n_numerical_features)


class PercentageOfNumericValuesWithMissingValues(MetaFeature[float]):
    """Percentage of numeric values with missing values."""

    dependencies = (NumericalColumns, NAValues)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        numerical_columns = NumericalColumns.retrieve(dependancy_values)
        if numerical_columns.empty:
            return 0.0

        na_values = NAValues.retrieve(dependancy_values)
        n_numerical_values = numerical_columns.shape[0] * numerical_columns.shape[1]
        n_numerical_values_with_missing_values = (
            na_values[numerical_columns.columns].sum().sum()
        )
        return float(n_numerical_values_with_missing_values / n_numerical_values)


class NumberOfNumericFeatures(MetaFeature[int]):
    """Number of numeric features in the dataset."""

    dependencies = (NumericalColumns,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> int:
        numerical_columns = NumericalColumns.retrieve(dependancy_values)
        if numerical_columns.empty:
            return 0

        return numerical_columns.shape[1]


class NumberOfCategoricalFeatures(MetaFeature[int]):
    """Number of categorical features in the dataset."""

    dependencies = (CategoricalColumns,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> int:
        categorical_columns = CategoricalColumns.retrieve(dependancy_values)
        if categorical_columns.empty:
            return 0

        return categorical_columns.shape[1]


class RatioNumericalToCategorical(MetaFeature[float]):
    """Ratio of numerical to categorical features in the dataset."""

    dependencies = (NumberOfNumericFeatures, NumberOfCategoricalFeatures)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        n_numerical = NumberOfNumericFeatures.retrieve(dependancy_values)
        n_categorical = NumberOfNumericFeatures.retrieve(dependancy_values)
        return float(n_numerical / n_categorical)


class RatioFeaturesToInstances(MetaFeature[float]):
    """Ratio of features to instances in the dataset."""

    dependencies = (NumberOfFeatures, InstanceCount)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        n_features = NumberOfFeatures.retrieve(dependancy_values)
        n_instances = InstanceCount.retrieve(dependancy_values)
        return float(n_features / n_instances)


class ClassCounts(DatasetStatistic[np.ndarray]):
    """Number of instances per class."""

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> np.ndarray:
        return np.unique(y, return_counts=True)[1]


class MinorityClassImbalance(MetaFeature[float]):
    """Imbalance of the minority class in the dataset. 0 => Balanced. 1 imbalanced."""

    dependencies = (ClassCounts,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        ratios, balanced_ratio = ClassImbalanceRatios.retrieve(dependancy_values)
        minority_ratio = ratios[0]
        distance_to_balanced = np.abs(minority_ratio - balanced_ratio)
        # This happens when all the ratio is in one class
        worst_case = (1 - len(ratios)) * balanced_ratio + (1 - balanced_ratio)
        return float(distance_to_balanced / worst_case)


class MajorityClassImbalance(MetaFeature[float]):
    """Imbalance of the majority class in the dataset. 0 => Balanced. 1 imbalanced."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        ratios, balanced_ratio = ClassImbalanceRatios.retrieve(dependancy_values)
        majority_ratio = ratios[-1]
        distance_to_balanced = np.abs(majority_ratio - balanced_ratio)
        # This happens when all the ratio is in one class
        worst_case = (1 - len(ratios)) * balanced_ratio + (1 - balanced_ratio)
        return float(distance_to_balanced / worst_case)


class ClassImbalance(MetaFeature[float]):
    """Mean Target Imbalance of the classes in general.

    0 => Balanced. 1 Imbalanced.
    """

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        ratios, balanced_ratio = ClassImbalanceRatios.retrieve(dependancy_values)
        return column_imbalance(ratios, balanced_ratio)


class ImbalancePerCategory(DatasetStatistic[Dict[str, float]]):
    """Imbalance of each categorical feature. 0 => Balanced. 1 most imbalanced.

    No categories implies perfectly balanced.
    """

    dependencies = ()

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> dict[str, float]:
        categoricals = x.select_dtypes(exclude=np.number)
        if categoricals.empty:
            return {}

        return {
            str(col): column_imbalance(*imbalance_ratios(x[col]))
            for col in categoricals.columns
        }


class MeanCategoricalImbalance(MetaFeature[float]):
    """The mean imbalance of categorical features."""

    dependencies = (ImbalancePerCategory,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        imbalances = ImbalancePerCategory.retrieve(dependancy_values)
        return float(np.mean(list(imbalances.values())))


class StdCategoricalImbalance(MetaFeature[float]):
    """The std imbalance of categorical features."""

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        imbalances = ImbalancePerCategory.retrieve(dependancy_values)
        return float(np.std(list(imbalances.values())))


class SkewnessPerNumericalColumn(DatasetStatistic[Dict[str, float]]):
    """Skewness of each numerical feature."""

    dependencies = (NumericalColumns,)

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> dict[str, float]:
        numericals = NumericalColumns.retrieve(dependancy_values)
        if numericals.empty:
            return {}

        skews: pd.Series = x.skew(numeric_only=True)  # type: ignore
        return dict(skews)


class SkewnessMean(MetaFeature[float]):
    """The mean skewness of numerical features."""

    dependencies = (SkewnessPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        skews = SkewnessPerNumericalColumn.retrieve(dependancy_values)
        if len(skews) == 0:
            return SKEWNESS_OF_NORMAL_DISTRIBUTION

        return float(np.mean(list(skews.values())))


class SkewnessStd(MetaFeature[float]):
    """The std skewness of numerical features."""

    dependencies = (SkewnessPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        skews = SkewnessPerNumericalColumn.retrieve(dependancy_values)
        if len(skews) == 0:
            return SKEWNESS_OF_NORMAL_DISTRIBUTION

        return float(np.std(list(skews.values())))


class SkewnessMin(MetaFeature[float]):
    """The min skewness of numerical features."""

    dependencies = (SkewnessPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        skews = SkewnessPerNumericalColumn.retrieve(dependancy_values)
        if len(skews) == 0:
            return SKEWNESS_OF_NORMAL_DISTRIBUTION

        return float(np.min(list(skews.values())))


class SkewnessMax(MetaFeature[float]):
    """The max skewness of numerical features."""

    dependencies = (SkewnessPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        skews = SkewnessPerNumericalColumn.retrieve(dependancy_values)
        if len(skews) == 0:
            return SKEWNESS_OF_NORMAL_DISTRIBUTION

        return float(np.max(list(skews.values())))


class KurtosisPerNumericalColumn(DatasetStatistic[Dict[str, float]]):
    """Kurtosis of each numerical feature."""

    dependencies = (NumericalColumns,)

    @override
    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series,
        dependancy_values: DSdict,
    ) -> dict[str, float]:
        numericals = NumericalColumns.retrieve(dependancy_values)
        if numericals.empty:
            return {}

        kurts: pd.Series = numericals.kurt()  # type: ignore
        return dict(kurts)


class KurtosisMean(MetaFeature[float]):
    """The mean kurtosis of numerical features."""

    dependencies = (KurtosisPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        kurts = KurtosisPerNumericalColumn.retrieve(dependancy_values)
        if len(kurts) == 0:
            return KURTOSIS_OF_NORMAL_DISTRIBUTION

        return float(np.mean(list(kurts.values())))


class KurtosisStd(MetaFeature[float]):
    """The std kurtosis of numerical features."""

    dependencies = (KurtosisPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        kurts = KurtosisPerNumericalColumn.retrieve(dependancy_values)
        if len(kurts) == 0:
            return KURTOSIS_OF_NORMAL_DISTRIBUTION

        return float(np.std(list(kurts.values())))


class KurtosisMin(MetaFeature[float]):
    """The min kurtosis of numerical features."""

    dependencies = (KurtosisPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        kurts = KurtosisPerNumericalColumn.retrieve(dependancy_values)
        if len(kurts) == 0:
            return KURTOSIS_OF_NORMAL_DISTRIBUTION

        return float(np.min(list(kurts.values())))


class KurtosisMax(MetaFeature[float]):
    """The max kurtosis of numerical features."""

    dependencies = (KurtosisPerNumericalColumn,)

    @override
    @classmethod
    def compute(cls, x: pd.DataFrame, y: pd.Series, dependancy_values: DSdict) -> float:
        kurts = KurtosisPerNumericalColumn.retrieve(dependancy_values)
        if len(kurts) == 0:
            return KURTOSIS_OF_NORMAL_DISTRIBUTION

        return float(np.max(list(kurts.values())))
