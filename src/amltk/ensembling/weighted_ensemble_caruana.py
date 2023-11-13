"""Implementation of the weighted ensemble procedure from Caruana et al. 2004.

???+ note "Reference"
    Ensemble selection from libraries of models

    Rich Caruana, Alexandru Niculescu-Mizil, Geoff Crew and Alex Ksikes

    ICML 2004

    https://dl.acm.org/doi/10.1145/1015330.1015432

    https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from amltk.randomness import as_rng

if TYPE_CHECKING:
    from amltk.types import Seed

logger = logging.getLogger(__name__)

# Values return by metric require that we can perform equality checks on them
T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


def weighted_ensemble_caruana(
    *,
    model_predictions: Mapping[K, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: Callable[[np.ndarray, np.ndarray], T],
    select: Callable[[Iterable[T]], T],
    seed: Seed | None = None,
) -> tuple[dict[K, float], list[tuple[K, T]], np.ndarray]:
    """Calculate a weighted ensemble of `n` models.

    Args:
        model_predictions: Mapping from model id to predictions
        targets: The targets
        size: The size of the ensemble to create
        metric: The metric to use in calculating which models to add to the ensemble.
        select: Selects a models from the list based on the values of the metric on
            their predictions. Can return a single ID or a list of them, in which
            case a random selection will be made.
        seed: The seed to use for breaking ties

    Returns:
        A mapping from id's to it's weight in the ensemble and the trajectory.
    """
    if not size > 0:
        raise ValueError("`size` must be positive")

    if len(model_predictions) == 0:
        raise ValueError("`model_predictions` is empty")

    rng = as_rng(seed)
    predictions = list(model_predictions.values())

    dtype = predictions[0].dtype
    if np.issubdtype(dtype, np.integer):
        logger.warning(
            f"Predictions were {dtype=}, converting to np.float64 to"
            " allow for weighted ensemble procedure.",
        )
        dtype = np.float64

    # Current sum of predictions in the ensemble
    current = np.zeros_like(predictions[0], dtype=dtype)

    # Buffer where new models predictions are added to current to try them
    buffer = np.empty_like(predictions[0], dtype=dtype)

    ensemble: list[K] = []
    trajectory: list[tuple[K, T]] = []

    def value_if_added(_pred: np.ndarray) -> T:
        # Get the value if the model was added to the current set of predicitons
        np.add(current, _pred, out=buffer)
        np.multiply(buffer, (1.0 / float(len(ensemble) + 1)), out=buffer)

        return metric(targets, buffer)

    for _ in range(size):
        # Get the value if added for each model
        scores = {_id: value_if_added(pred) for _id, pred in model_predictions.items()}

        # Get the choices that produce the best value
        chosen_val = select(scores.values())

        choices = [_id for _id, score in scores.items() if score == chosen_val]
        choice = rng.choice(np.asarray(choices))

        # Add the predictions of the chosen model
        np.add(current, model_predictions[choice], out=current)

        # Record it's addition and the score of the ensemble with this
        # choice added
        ensemble.append(choice)
        trajectory.append((choice, chosen_val))

        # In the case of only one model, have calculated it's loss
        # and it's the only available model to add to the ensemble
        if len(model_predictions) == 1:
            ensemble *= size
            trajectory *= size
            break

    final = np.multiply(current, (1.0 / float(len(ensemble))))

    return (
        {_id: count / size for _id, count in Counter(ensemble).items()},
        trajectory,
        final,
    )
