"""Utilities for voting ensembles."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils import Bunch

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

_Voter = TypeVar("_Voter", VotingRegressor, VotingClassifier)


def voting_with_preffited_estimators(
    estimators: Iterable[BaseEstimator],
    weights: Iterable[float] | None = None,
    *,
    voter: type[_Voter] | None = None,
    **voting_kwargs: Any,
) -> _Voter:
    """Create a voting ensemble with pre-fitted estimators.

    Args:
        estimators: The estimators to use in the ensemble.
        weights: The weights to use for the estimators. If None,
            will use uniform weights.
        voter: The voting classifier or regressor to use.
            If None, will use the appropriate one based on the type of the first
            estimator.
        **voting_kwargs: Additional arguments to pass to the voting classifier or
            regressor.

    Returns:
        The voting classifier or regressor with the pre-fitted estimators.
    """
    estimators = list(estimators)
    est0 = estimators[0]
    is_classification = voter is not None and issubclass(voter, VotingClassifier)

    if voter is None:
        if isinstance(est0, ClassifierMixin):
            voter_cls = VotingClassifier
            is_classification = True
        elif isinstance(est0, ClassifierMixin):
            voter_cls = VotingRegressor
            is_classification = False
        else:
            raise ValueError(
                f"Could not infer voter type from estimator type: {type(est0)}."
                " Please specify the voter type explicitly.",
            )
    else:
        voter_cls = voter

    if weights is None:
        weights = np.ones(len(estimators)) / len(estimators)
    else:
        weights = list(weights)

    named_estimators = [(str(i), e) for i, e in enumerate(estimators)]
    _voter = voter_cls(named_estimators, weights=weights, **voting_kwargs)
    _voter.estimators_ = [model for _, model in _voter.estimators]  # type: ignore

    if is_classification:
        est0_classes_ = est0.classes_  # type: ignore
        _voter.classes_ = est0_classes_  # type: ignore
        if np.ndim(est0_classes_) > 1:
            est0_classes_ = est0_classes_[0]
            _voter.le_ = MultiLabelBinarizer().fit(est0_classes_)  # type: ignore
        else:
            _voter.le_ = LabelEncoder().fit(est0.classes_)  # type: ignore

    _voter.named_estimators_ = Bunch()  # type: ignore

    # Taken from Sklearn _BaseVoting.fit
    # Uses 'drop' as placeholder for dropped estimators
    est_iter = iter(_voter.estimators_)  # type: ignore
    for name, est in _voter.estimators:  # type: ignore
        current_est = est if est == "drop" else next(est_iter)
        _voter.named_estimators_[name] = current_est  # type: ignore

        if hasattr(current_est, "feature_names_in_"):
            _voter.feature_names_in_ = current_est.feature_names_in_  # type: ignore

    return _voter  # type: ignore
