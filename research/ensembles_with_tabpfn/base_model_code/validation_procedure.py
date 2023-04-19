from typing import List

import numpy as np
import pandas as pd

from statistics import mean

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.base import clone


def _agg_fold_model_predictions(oof_per_split, oof_template_generator):
    """Aggregates the predictions of the models trained on a specific fold.

        Such a "complicated" aggregation is needed as not every fold model predicts for every sample.
        Hence, some samples are nan for some fold models. This function takes care of this.

        In essence, the returned predictions is the average over the different repetitions.
    """

    # -- Avg oof
    oof_out = oof_template_generator()
    pred_count = np.full((oof_out.shape[0],), 0)  # could also just count it, but this is saver if some splits fail?
    for oof in oof_per_split:
        pred_count = pred_count + ~np.isnan(oof).any(axis=1)
        # following https://stackoverflow.com/a/50642947
        oof_out = np.where(
            np.isnan(oof) & np.isnan(oof_out),
            np.nan,
            np.nansum(np.stack((oof, oof_out)), axis=0)
        )

    # - Get average
    oof_out = oof_out / pred_count[:, None]
    # - Normalize
    oof_out = oof_out / oof_out.sum(axis=1)[:, None]

    return oof_out


def predict_fit_repeated_cross_validation(n: int, k: int, input_model: sklearn_Pipeline,
                                          X_train: pd.DataFrame, y_train: np.array,
                                          X_test: pd.DataFrame, y_test: np.array,
                                          metric_data: dict):
    """ Code to for n-repeated k-fold cross-validation.
        -> Computes validation data (average-over-repeats OOF), validation score, and test predictions.

        Note:
            - we assume y to be encoded (with a label encoder)

        TODO:
            - we likely want to parallelize this as well or only parallelize the model's training...

    Parameters
    ----------
    n: int
        Number of repeats
    k: int
        Number of folds
    input_model: sklearn_Pipeline
        sklearn pipeline to fit and predict with
    X_train: pd.Dataframe
        Training data
    y_train: np.array
        Training labels
    X_test: pd.Dataframe
        Test data
    y_test: np.array
        Test labels
    metric_data: dict
        contains information on the metric and its task type:
            * function: callable    # metric function
            * requires_proba: bool  # determines if predict or predict_proba is used
            * task_type: str    # classification or regression
    """
    cv = list(RepeatedStratifiedKFold(n_repeats=n, n_splits=k, random_state=0).split(X_train, y_train))
    is_classification = metric_data["task_type"] == "classification"

    if is_classification:
        n_classes = np.unique(np.concatenate([np.unique(y_train), np.unique(y_test)])).shape[0]
        oof_template_generator = lambda: np.full((X_train.shape[0], n_classes), np.nan)

        if metric_data["requires_proba"]:
            metric = metric_data["function_proba_in"]
        else:
            metric = metric_data["function"]
    else:
        # oof_template_generator = lambda: np.full((X_train.shape[0],), np.nan)
        raise NotImplementedError("Regression not yet implemented")

    # -- Init
    oof_per_split = []  # type: List[np.ndarray]
    train_cv_scores = []  # type: List[float]
    val_cv_scores = []  # type: List[float]
    test_predictions_per_fold_model = []  # type: List[np.ndarray]

    # -- Compute values for each split
    for train_index, test_index in cv:
        # - Fold data
        fold_X_train, fold_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        # -- Fit and predict with fold model (we call it a fold model could also be understood as a split model)
        # copy/clone to avoid anything that might be consistent across fits
        #  TODO: determine what happens to the randomstate of the model here...
        #       technically need to have a different one for each split or a random state for across splits...
        fold_model = clone(input_model)
        fold_model.fit(fold_X_train, fold_y_train)

        # -- Predict on fold train data (i.e., train score / reproduction score)
        # (not the true reproduction score if this is a stacking model!)
        if metric_data["requires_proba"]:
            train_score = metric(fold_y_train, fold_model.predict_proba(fold_X_train))
        else:
            train_score = metric(fold_y_train, fold_model.predict(fold_X_train))
        train_cv_scores.append(train_score)

        # -- Predict on fold test data (i.e., validation data)
        fold_y_pred_proba = None
        if metric_data["requires_proba"]:
            fold_y_pred_proba = fold_model.predict_proba(fold_X_test)
            val_score = metric(fold_y_test, fold_y_pred_proba)
        else:
            val_score = metric(fold_y_test, fold_model.predict(fold_X_test))
        val_cv_scores.append(val_score)

        # -- Predict on outer test
        test_predictions_per_fold_model.append(fold_model.predict_proba(X_test))

        # -- Get fold_y_pred_proba if needed
        # TODO: maybe add check that proba and predictions match...
        if fold_y_pred_proba is None:
            fold_y_pred_proba = fold_model.predict_proba(fold_X_test)

        # -- Save oof
        oof = oof_template_generator()
        oof[test_index] = fold_y_pred_proba
        oof_per_split.append(oof)

    # -- Avg scores
    train_score = mean(train_cv_scores)
    val_score = mean(val_cv_scores)

    if is_classification:
        # -- Val data
        # TODO: technically ignores model specific threshold for non-proba models...
        #   Would need to do a majority vote for label predictions otherwise...?
        #   Do we really want to do that?
        #   Determine best threshold usage...
        val_probabilities = _agg_fold_model_predictions(oof_per_split, oof_template_generator)
        val_predictions = np.argmax(val_probabilities, axis=1)

        # -- Test data
        # - Memory efficient average
        test_probabilities = np.full(test_predictions_per_fold_model[0].shape, 0, dtype="float64")
        for fold_model_predictions in test_predictions_per_fold_model:
            np.add(test_probabilities, fold_model_predictions, out=test_probabilities)
        test_probabilities /= len(test_predictions_per_fold_model)
        # - Normalize to guarantee sum to 1
        test_probabilities = test_probabilities / test_probabilities.sum(axis=1)[:, None]
        test_predictions = np.argmax(test_probabilities, axis=1)

        if metric_data["requires_proba"]:
            test_score = metric(y_test, test_probabilities)
        else:
            test_score = metric(y_test, test_predictions)
    else:
        raise NotImplementedError

    return train_score, val_score, test_score, val_probabilities, val_predictions, test_probabilities, test_predictions

