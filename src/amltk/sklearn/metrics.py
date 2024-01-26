"""Utilities for sklearn metrics."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _BaseScorer

from amltk.optimization.metric import Metric

# All of these bounds are from the perspective of an sklearn **scorer**
# where the are already negated
# these metrics are taken from `get_scorer`
_SCORER_BOUNDS = {
    "explained_variance": (-np.inf, 1.0),  # Default metric is positive
    "r2": (-np.inf, 1.0),  # Default metric is positive
    "max_error": (-np.inf, 0),  # Default metric is negative
    "matthews_corrcoef": (-1.0, 1.0),  # Default metric is positive
    "neg_median_absolute_error": (-np.inf, 0),  # Default metric is negative
    "neg_mean_absolute_error": (-np.inf, 0),  # Default metric is negative
    "neg_mean_absolute_percentage_error": (-np.inf, 0),  # Default metric is negative
    "neg_mean_squared_error": (-np.inf, 0),  # Default metric is negative
    "neg_mean_squared_log_error": (-np.inf, 0),  # Default metric is negative
    "neg_root_mean_squared_error": (-np.inf, 0),  # Default metric is negative
    "neg_root_mean_squared_log_error": (-np.inf, 0),  # Default metric is negative
    "neg_mean_poisson_deviance": (-np.inf, 0),  # Default metric is negative
    "neg_mean_gamma_deviance": (-np.inf, 0),  # Default metric is negative
    "accuracy": (0, 1.0),  # Default metric is positive
    "top_k_accuracy": (0, 1.0),  # Default metric is positive
    "roc_auc": (0, 1.0),  # Default metric is positive
    "roc_auc_ovr": (0, 1.0),  # Default metric is positive
    "roc_auc_ovo": (0, 1.0),  # Default metric is positive
    "roc_auc_ovr_weighted": (0, 1.0),  # Default metric is positive
    "roc_auc_ovo_weighted": (0, 1.0),  # Default metric is positive
    "balanced_accuracy": (0, 1.0),  # Default metric is positive
    "average_precision": (0, 1.0),  # Default metric is positive
    "neg_log_loss": (-np.inf, 0),  # Default metric is negative
    "neg_brier_score": (-np.inf, 0),  # Default metric is negative
    "positive_likelihood_ratio": (0, np.inf),  # Default metric is positive
    "neg_negative_likelihood_ratio": (-np.inf, 0),  # Default metric is negative
    # Cluster metrics that use supervised evaluation
    "adjusted_rand_score": (-0.5, 1.0),  # Default metric is positive
    "rand_score": (0, 1.0),  # Default metric is positive
    "homogeneity_score": (0, 1.0),  # Default metric is positive
    "completeness_score": (0, 1.0),  # Default metric is positive
    "v_measure_score": (0, 1.0),  # Default metric is positive
    "mutual_info_score": (0, np.inf),  # Default metric is positive
    # TODO: Not sure about the lower bound on this.
    # Seems that 0 is pure randomness but theoretically it could be negative
    "adjusted_mutual_info_score": (-1.0, 1.0),  # Default metric is positive
    "normalized_mutual_info_score": (0.0, 1.0),  # Default metric is positive
    "fowlkes_mallows_score": (0.0, 1.0),  # Default metric is positive
}


def as_metric(
    scorer: str | _BaseScorer,
    *,
    bounds: tuple[float, float] | None = None,
    name: str | None = None,
) -> Metric:
    """Convert a scorer to a metric."""
    match scorer:
        case str():
            _scorer = get_scorer(scorer)
            _name = scorer if name is None else name
        case _BaseScorer():
            _scorer = scorer
            if name is not None:
                _name = name
            else:
                _name = scorer._score_func.__name__
                _name = f"neg_{_name}" if scorer._sign == -1 else _name
        case _:
            raise TypeError(f"Cannot convert {scorer!r} to a metric.")

    # This is using what sklearn use in their `__repr__` method
    if bounds is None:
        bounds = _SCORER_BOUNDS.get(_name, None)

    if bounds is None:
        warnings.warn(
            f"Cannot infer bounds for scorer {_name}. Please explicitly provide "
            " them with the `bounds` argument or set them to `(-np.inf, np.inf)`.",
            UserWarning,
            stacklevel=2,
        )

    # Sklearn scorers are always positive
    return Metric(name=_name, bounds=bounds, minimize=False)
