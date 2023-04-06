# Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)
#   Original take with (heavy) adaptions to be usable from https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/abstract_ensemble.py
#   This version is without passthrough (for stacking)
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


class AbstractEnsemble(object):
    """Abstract ensemble class

    During fit (for classification), we transform all labels into integers. This also happens for the predictions of
    base models.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models. Base models must have label encoders for the data they have been fit on.
    predict_method_base_models: {"predict", "predict_proba"}, default="predict"
        Determine the predict method that is used to obtain the output of base models that is passed to an ensemble's
        fit method.
    predict_method_ensemble_predict: {"predict", "predict_proba", "dynamic", None}, default=None
        Determine the predict method that is used to obtain the output of base models that is passed to an ensemble's
        predict method.
            * None, the same method passed to ensemble fit is passed to ensemble predict.
            * "dynamic", the method is selected based on the predict or predict_proba call.

    Attributes
    ----------
    le_ : LabelEncoder, object
        The label encoder created at :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, base_models, predict_method_base_models: str = "predict",
                 predict_method_ensemble_predict: Optional[str] = None):

        self.base_models = base_models
        self.predict_method = predict_method_base_models

        # Get the classes seen by the base model on the data they have been trained on.
        self.base_model_le_ = self.base_models[0].le_

        if predict_method_ensemble_predict is None:
            self.predict_method_ensemble_predict = predict_method_base_models
        else:

            if predict_method_ensemble_predict not in ["predict", "predict_proba", "dynamic"]:
                raise ValueError("The value for predict_method_ensemble_predict is not allowed.")

            self.predict_method_ensemble_predict = predict_method_ensemble_predict

    def fit(self, X, y):
        """Fitting the ensemble. To do so, we get the predictions of the base models and pass it to the ensemble's fit.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns
        """
        if self.predict_method not in ["predict", "predict_proba"]:
            raise ValueError("Unknown predict method: {}".format(self.predict_method))

        # TODO: remove none and assume that input data is preprocessed to be only numeric!
        #   - Fair assumption if this is part of the pipeline; otherwise only needed if passthrough is used.
        X, y = check_X_y(X, y, dtype=None)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        # Check if self.classes_ differs
        if len(self.classes_) != len(self.base_model_le_.classes_):
            warnings.warn("The number of seen classes differs for the base models and the ensemble." +
                          "We fix this by using the base model's label encoder.")
            # To fix it, we use the label encoder of the base models
            self.le_ = self.base_model_le_
            self.classes_ = self.base_model_le_.classes_

        y_ = self.le_.transform(y)

        self.ensemble_fit(self.base_models_predictions(X), y_)

        return self

    def predict(self, X):
        """Predicting with the ensemble. To do so, we get the predictions of the base models and pass it to the ensemble

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        check_is_fitted_for = []
        if hasattr(self, "fitted_attributes"):
            check_is_fitted_for = self.fitted_attributes

        check_is_fitted(self, ["le_", "classes_"] + check_is_fitted_for)
        # TODO: as for fit
        X = check_array(X, dtype=None)
        bm_preds = self.base_models_predictions_for_ensemble_predict(X)

        ensemble_output = self.ensemble_predict(bm_preds)

        return self.le_.inverse_transform(ensemble_output)

    def predict_proba(self, X):
        """Predicting with the ensemble.
         To do so, we get the predictions of the base models and pass it to the ensemble

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            Vector containing the class probabilities for each class for each sample.
        """
        check_is_fitted(self, ["le_", "classes_"])
        # TODO: as for fit
        X = check_array(X, dtype=None)
        bm_preds = self.base_models_predictions_for_ensemble_predict(X, predict_for="predict_proba")

        ensemble_output = self.ensemble_predict_proba(bm_preds)

        return ensemble_output

    def base_models_predictions(self, X):
        if self.predict_method == "predict":
            return [self.le_.transform(bm.predict(X)) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    @staticmethod
    def _confidences_to_predictions(confidences):
        # TODO: implement custom threshold usage here
        return np.argmax(confidences, axis=1)

    def base_models_predictions_for_ensemble_predict(self, X, predict_for="predict"):
        # Catch dynamic case
        if self.predict_method_ensemble_predict == "dynamic":
            pred_method = predict_for
        else:
            pred_method = self.predict_method_ensemble_predict

        if pred_method == "predict":
            return [self.le_.transform(bm.predict(X)) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    @abstractmethod
    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> 'AbstractEnsemble':
        """Fit an ensemble given predictions of base models and targets.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        labels : array of shape [n_targets]

        Returns
        -------
        self

        """
        pass

    @abstractmethod
    def ensemble_predict(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        pass

    def ensemble_predict_proba(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """Create ensemble probability predictions from the base model predictions.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification

        Returns
        -------
        y : ndarray, shape (n_samples, n_targets)
            Vector containing the class probabilities for each sample.
        """
        raise NotImplemented("ensemble_predict_proba is not implemented by every ensemble method!")
