import numpy as np
from typing import List

from research.ensembles_with_tabpfn.complementary_code.data_handler import \
    FakedFittedAndValidatedClassificationBaseModel as BaseModel

EPS = np.finfo(float).eps


def _normalized_geometric_mean(prediction_probabilities):
    m = len(prediction_probabilities)
    mean_preds = np.ones_like(prediction_probabilities[0])

    # Memory efficient
    for bm_preds in prediction_probabilities:
        np.multiply(mean_preds,
                    # EPS to avoid zeros
                    (bm_preds ** (1 / m)) + EPS,
                    out=mean_preds)

    return mean_preds / mean_preds.sum(axis=1)[:, None]


def _kl_divergence(p, q):
    # Following: https://datascience.stackexchange.com/a/26318
    # modified to return the expectation over all instances

    p += EPS
    q += EPS

    return np.mean(np.sum(p * np.log(p / q), axis=1))


def _mean_kl_for_ngm_preds(bm_prediction_probabilities):
    # A sample of the diversity for cross-entropy over all instance for the predictions.
    # The result of this must be averaged over multiple samples of different sources of
    #   randomness for diversity (but not for disparity where the expectation is taken earlier).

    ensemble_combiner = _normalized_geometric_mean(bm_prediction_probabilities)

    # No need for memory efficient average
    div_per_model = [_kl_divergence(ensemble_combiner, bm_pred) for bm_pred in bm_prediction_probabilities]
    return sum(div_per_model) / len(div_per_model)


def ensemble_diversity_analysis(y_val, y_test, base_models: List[BaseModel], complement_model: BaseModel):
    """Analysis diversity for cross-entropy loss, following Wood et al. 2023

        Following this framework, we are not dependent on the label.
        I supply it here if we want to use other metrics that are in the future.
    """

    bm_predictions = [bm.val_probabilities for bm in base_models]

    # Diversity (based on cross-entropy loss)
    bm_div = _mean_kl_for_ngm_preds(bm_predictions)
    bm_c_div = _mean_kl_for_ngm_preds(bm_predictions + [complement_model.val_probabilities])

    return bm_div, bm_c_div


def compute_left_bergman_centroid(predictions_of_base_model):
    """Assumes as input the predictions of a base model over multiple sources of randomness (for cross-entropy)"""

    mean_preds = np.zeros_like(predictions_of_base_model[0])

    # Memory efficient
    for algo_bm_preds in predictions_of_base_model:
        np.add(mean_preds,
               # EPS to avoid -inf
               np.log(algo_bm_preds + EPS),
               out=mean_preds)

    mean_preds /= len(predictions_of_base_model)
    mean_preds = np.exp(mean_preds)

    return mean_preds / mean_preds.sum(axis=1)[:, None]


def ensemble_disparity_analysis(lbc_base_models_predictions, lbc_complement_model_predictions):
    """Analysis disparity for cross-entropy loss, following Wood et al. 2023"""

    bm_disp = _mean_kl_for_ngm_preds(lbc_base_models_predictions)
    bm_c_disp = _mean_kl_for_ngm_preds(lbc_base_models_predictions + [lbc_complement_model_predictions])

    return bm_disp, bm_c_disp
