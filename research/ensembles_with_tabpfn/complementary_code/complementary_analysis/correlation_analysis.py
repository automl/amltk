import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelBinarizer
import pandas as pd


# -- Correlation related code
def _get_transformed_labels(y_true):
    # -- Obtain transformed labels
    lb = LabelBinarizer()
    lb.fit(y_true)
    transformed_labels = lb.transform(y_true)

    # Fix for binary
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)

    return transformed_labels


def _compute_loss_correlation(transformed_labels, proba_a, proba_b):
    proba_a_loss = 1 - (transformed_labels * proba_a).sum(axis=1)
    proba_b_loss = 1 - (transformed_labels * proba_b).sum(axis=1)

    if np.array_equal(proba_a, proba_b_loss):
        return 1.0

    return pearsonr(proba_a_loss, proba_b_loss)[0]


def _compute_correlation_matrix(transformed_labels, base_models, complement_model_name):
    loss_per_bm = [1 - (transformed_labels * bm.val_probabilities).sum(axis=1) for bm in base_models]
    corr_matrix = np.abs(np.corrcoef(loss_per_bm))

    # Replace nan values with 1, nan occurs when two compared vectors are identical;
    corr_matrix = np.nan_to_num(corr_matrix, nan=1)

    algo_names_list = [bm.config["algorithm"] for bm in base_models]
    algo_names_u = list(set(algo_names_list))
    algo_names_list = np.array(algo_names_list)

    # Aggregate over different base models of same algorithm
    res_list = []
    for algo_name in algo_names_u:
        res_list.append(np.mean(corr_matrix[:, algo_names_list == algo_name], axis=1))

    corr_matrix = np.array(res_list).T
    res_list = []
    for algo_name in algo_names_u:
        res_list.append(np.mean(corr_matrix[algo_names_list == algo_name, :], axis=0))
    corr_matrix = np.array(res_list)

    # Reorder
    tmp_df = pd.DataFrame(corr_matrix, columns=algo_names_u, index=algo_names_u)
    sorted_index = sorted(algo_names_u)
    sorted_index.remove(complement_model_name)
    sorted_index += [complement_model_name]
    tmp_df = tmp_df.loc[sorted_index, sorted_index]
    corr_matrix = tmp_df.to_numpy()
    # Filter
    corr_matrix[np.tril_indices_from(corr_matrix, k=-1)] = -1

    return pd.DataFrame(corr_matrix, columns=tmp_df.columns, index=tmp_df.index)


def correlation_analysis(y_val, y_test, ens_bm_predictions, ens_bm_with_complement_predictions, base_models,
                         complement_algorithm_name):
    ens_prediction_correlation = _compute_loss_correlation(_get_transformed_labels(y_test), ens_bm_predictions,
                                                           ens_bm_with_complement_predictions)

    all_base_models_correlation_df = _compute_correlation_matrix(_get_transformed_labels(y_val), base_models,
                                                                 complement_algorithm_name)

    return ens_prediction_correlation, all_base_models_correlation_df
