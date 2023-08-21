"""Extracts metafeatures from a dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from amltk.metalearning.metafeatures import DatasetStatistic, MetaFeature

logger = logging.getLogger(__name__)


@dataclass
class MetaFeatureExtractor:
    """Extracts metafeatures from a dataset.

    Args:
        metafeatures: A sequence of metafeatures to extract. By
            default, all metafeatures will be extracted.
    """

    metafeatures: Sequence[type[DatasetStatistic]] | None = None

    def __call__(
        self,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        """Extracts metafeatures from a dataset.

        Args:
            x: The dataset's features.
            y: The dataset's target.

        Returns:
            A dataframe of metafeatures and their values.
        """
        return pd.Series(
            {
                key.name(): value
                for key, value in self.raw_values(x=x, y=y).items()
                if issubclass(key, MetaFeature)
            },
        )

    def _calc(
        self,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        metafeature: type[DatasetStatistic],
        values: dict[type[DatasetStatistic], Any],
    ) -> dict[type[DatasetStatistic], Any]:
        """Calculates a metafeature's value.

        !!! warning "Updates values in-place"

            This function updates the `values` dictionary in-place.

        Args:
            x: The dataset's features.
            y: The dataset's target.
            metafeature: The metafeature to calculate.
            values: A dictionary of metafeatures to statistics and their values.

        Returns:
            A dictionary of metafeatures and dataset statistics and their values.
        """
        for dep in metafeature.dependencies:
            values = self._calc(
                x=x,
                y=y,
                metafeature=dep,
                values=values,
            )

        if metafeature not in values:
            values[metafeature] = metafeature.compute(
                x=x,
                y=y,
                dependancy_values=values,
            )
        return values

    def raw_values(
        self,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
    ) -> dict[type[DatasetStatistic], Any]:
        """Extracts metafeatures from a dataset.

        Args:
            x: The dataset's features.
            y: The dataset's target.

        Returns:
            A dictionary of metafeatures and dataset statistics and their values.
        """
        if self.metafeatures is None:
            from amltk.metalearning import get_metafeatures

            self.metafeatures = get_metafeatures()

        values: dict[type[DatasetStatistic], Any] = {}
        for metafeature in self.metafeatures:
            values = self._calc(
                x=x,
                y=y,
                metafeature=metafeature,
                values=values,
            )

        return values
