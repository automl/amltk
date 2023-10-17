"""Custom estimators for use with scikit-learn."""

# TODO:
#   * Document Stored Prediction Estimators
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import Self

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from amltk.data.conversions import probabilities_to_classes

if TYPE_CHECKING:
    import numpy as np


class StoredPredictionRegressor(BaseEstimator, RegressorMixin):
    """A class that just uses precomputed values for regression."""

    def __init__(self, predictions: np.ndarray):
        """Initialize the estimator.

        Args:
            predictions: The precomputed predictions.
        """
        super().__init__()
        self.predictions = predictions

    def fit(self, *_: Any, **__: Any) -> Self:
        """Fit the estimator. Doesn't do anything."""
        return self

    def predict(self, X: Any, *_: Any, **__: Any) -> np.ndarray:  # noqa: N803, ARG002
        """Predict the target values, returning the precomputed values."""
        return self.predictions


class StoredPredictionClassifier(BaseEstimator, ClassifierMixin):
    """A class that just uses precomputed values for classification."""

    def __init__(
        self,
        predictions: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
        classes: list[np.ndarray] | np.ndarray | None = None,
    ):
        """Initialize the estimator.

        Args:
            predictions: The precomputed predictions, if any.
            probabilities: The precomputed probabilities, if any.
            classes: The classes, if any.
        """
        super().__init__()
        self.predictions = predictions
        self.probabilities = probabilities
        self.classes = classes

        # HACK: This is to enable sklearn-compatibility
        # `clone` and other methods rely on this trailing underscore
        # to indicate fitted attributes. We essentially declare it fitted
        # at init for simplicity
        self.classes_ = classes

    def fit(self, *_: Any, **__: Any) -> Self:
        """Fit the estimator. Doesn't do anything."""
        return self

    def predict(self, X: Any, *_: Any, **__: Any) -> np.ndarray:  # noqa: N803, ARG002
        """Predict the target values, returning the precomputed values."""
        if self.predictions is None:
            if self.probabilities is None:
                raise RuntimeError(
                    "No predictions or probabilities were provided during",
                    " construction, so this estimator cannot be used for",
                    " `predict()`.",
                )
            if self.classes_ is None:
                raise RuntimeError(
                    "No classes were provided during construction, so it can't"
                    " be used for `predict()` from probabilities.",
                )

            predictions = probabilities_to_classes(
                self.probabilities,
                classes=self.classes_,
            )
        else:
            predictions = self.predictions

        return predictions

    def predict_proba(
        self,
        X: Any,  # noqa: N803, ARG002
        *_: Any,
        **__: Any,
    ) -> np.ndarray:
        """Predict the probabilities, returning the precomputed values."""
        if self.probabilities is None:
            raise RuntimeError(
                "No probabilities were provided during construction, so this"
                " estimator cannot be used for `predict_proba()`.",
            )

        return self.probabilities
