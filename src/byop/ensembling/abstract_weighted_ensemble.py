# Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)
from abc import ABCMeta, abstractmethod
from .abstract_ensemble import AbstractEnsemble
from typing import List, Union
import numpy as np
from math import isclose


class AbstractWeightedEnsemble(AbstractEnsemble):
    __metaclass__ = ABCMeta

    def __init__(self, *args):
        super().__init__(*args)
        self.normalize_predict_proba_ = False

    @staticmethod
    def _calculate_weights(discrete_weights) -> np.ndarray:
        weights = discrete_weights / np.sum(discrete_weights)

        return weights

    @staticmethod
    def _calculate_counts(float_weights, sample_size) -> np.ndarray:
        weights = float_weights * sample_size

        return weights

    @staticmethod
    def _ensemble_predict(predictions: List[np.ndarray], weights: np.ndarray,
                          normalize_predict_proba: bool = False) -> np.ndarray:
        """Blanket (not-the-most-efficient) ensemble predict for a weighted ensemble.

        Parameters
        ----------
        weights: np.ndarray
            Can be of any numeric range (but some metric might require normalization).
        normalize_predict_proba: bool, default=False
            If True, normalize the prediction probabilities such that they sum up to 1 and are in [0,1].
            Only needed if the weights are not in [0,1] or do not sum up to 1.
            We apply the softmax but only if negative weights are present.
            We apply simple normalization if weights are positive but do not sum to 1.
        """

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        for pred, weight in zip(predictions, weights):
            np.multiply(pred, weight, out=tmp_predictions)
            np.add(average, tmp_predictions, out=average)

        if normalize_predict_proba:
            if any(weights < 0):
                exp = np.nan_to_num(np.exp(np.clip(average, -88.72, 88.72)))  # Clip to avoid overflow
                average = exp / exp.sum(axis=1)[:, None]
                average = average / average.sum(axis=1)[:, None]
            elif not isclose(weights.sum(), 1):
                average = average / average.sum(axis=1)[:, None]

        return average

    @abstractmethod
    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> 'AbstractEnsemble':
        pass

    @property
    def attr_after_fit(self):
        return ["weights_", "validation_loss_", "iteration_batch_size_", "val_loss_over_iterations_"]

    @property
    def is_not_fitted(self) -> List[str]:
        """Returns a list of missing attributes or empty list if none are missing."""
        missing_attr = [req_attr for req_attr in self.attr_after_fit if not hasattr(self, req_attr)]
        return missing_attr

    def ensemble_predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        if self.is_not_fitted:
            raise ValueError("Model not fitted! Missing fit parameters are:", self.is_not_fitted)

        return self._confidences_to_predictions(self.ensemble_predict_proba(predictions))

    def ensemble_predict_proba(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        if self.is_not_fitted:
            raise ValueError("Model not fitted! Missing fit parameters are:", self.is_not_fitted)

        if len(predictions) == len(self.weights_):
            # predictions include those of zero-weight models.
            return self._ensemble_predict(predictions, self.weights_,
                                          normalize_predict_proba=self.normalize_predict_proba_)

        elif len(predictions) == np.count_nonzero(self.weights_):
            # predictions do not include those of zero-weight models.
            non_null_weights = self.weights_[self.weights_ != 0]
            return self._ensemble_predict(predictions, non_null_weights,
                                          normalize_predict_proba=self.normalize_predict_proba_)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")

    @property
    def _to_save_metadata(self):
        """Standard metadata for weighted ensemble

        Parameters
        -------
        self.weights_: np.ndarray
            The best found weights
        self.validation_loss_: float
            The validation loss of the best found weights
        self.iteration_batch_size_: int
            The batch size used in each iteration of the selection method.
            This is equal to the number of evolutions that are done inbetween checking whether
            we found a new best weight vector.
        self.val_loss_over_iterations_: List[float]
            The best validation loss over each iteration of the search.
        """

        # Need to transform any numpy datatypes to normal python dtypes such that we can use JSON
        metadata = dict(
            weights_=self.weights_.tolist(),
            validation_loss_=float(self.validation_loss_),
            iteration_batch_size_=int(self.iteration_batch_size_),
            val_loss_over_iterations_=np.array(self.val_loss_over_iterations_).tolist()
        )
        metadata.update(self.model_specific_metadata_)

        return metadata
