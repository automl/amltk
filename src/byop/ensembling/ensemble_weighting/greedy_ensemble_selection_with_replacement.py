# Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)
#   Original take with (heavy) adaptions to be usable from  https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py
import os
import warnings
from collections import Counter
from typing import List, Optional, Union, Callable
import logging

import numpy as np

from sklearn.utils import check_random_state
from ..abstract_weighted_ensemble import AbstractWeightedEnsemble

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class GreedyEnsembleSelection(AbstractWeightedEnsemble):
    """A weighted mean ensemble following the greedy procedure from Caruana et al. 2004.

    Fitting an EnsembleSelection generates an ensemble from the models
    generated during the search process. Can be further used for prediction.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    n_iterations: int
        The number of iterations that the algorithm shall run. TODO: replace by time limit?
        This results in `n_iterations * len(base_models)` many metric function evaluations in total.
    loss_function: Callable
        TODO: implement this using a scorer (if possible)
        A function that maps (prediction_probabilities, labels) -> loss.
    n_jobs: int, default=-1
        Number of processes to use for parallelization. -1 means all available.
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection. This algorithm requires randomness to break ties.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    """

    def __init__(self, base_models: List[object], n_iterations: int, loss_function: Callable,
                 n_jobs: int = -1, random_state: Optional[Union[int, np.random.RandomState]] = None) -> None:

        super().__init__(base_models, "predict_proba")
        self.n_iterations = int(n_iterations)
        self.loss_function = loss_function
        self.num_input_models_ = len(base_models)

        # -- Code for multiprocessing
        if (n_jobs == 1) or (os.name == "nt"):
            self._use_mp = False
            # FIXME: Found no way to make this work on windows with the multiprocessing below
            if os.name == "nt":
                warnings.warn("WARNING: n_jobs != 1 is not supported on Windows. Setting n_jobs=1.")
        else:
            if n_jobs == -1:
                n_jobs = len(os.sched_getaffinity(0))
            self._n_jobs = n_jobs
            self._use_mp = True

        # -- Randomness Code
        self.random_state = random_state

    def ensemble_fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> AbstractWeightedEnsemble:
        if self.n_iterations < 1:
            raise ValueError('Number of iterations cannot be less than one!')

        self._fit(predictions, labels)
        self._apply_use_best()
        self._calculate_final_weights()

        # -- Set metadata correctly
        self.iteration_batch_size_ = len(predictions)  # Batch size is equal to the number of base models

        return self

    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> None:
        """Caruana's ensemble selection with replacement."""

        # -- Init Vars
        ensemble = []  # type: List[np.ndarray]
        trajectory = []  # contains iteration best
        self.val_loss_over_iterations_ = []  # contains overall best
        order = []
        rand = check_random_state(self.random_state)
        weighted_ensemble_prediction = np.zeros(predictions[0].shape, dtype=np.float64)

        # -- Init Memory-efficient averaging
        if not self._use_mp:
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape, dtype=np.float64)

        for i in range(self.n_iterations):
            logger.debug(f"Iteration {i}")

            ens_size = len(ensemble)
            if ens_size > 0:
                np.add(weighted_ensemble_prediction, ensemble[-1], out=weighted_ensemble_prediction)

            # -- Process Iteration Solutions
            if self._use_mp:
                losses = self._compute_losses_mp(weighted_ensemble_prediction, labels, predictions, ens_size)
            else:
                losses = np.zeros((len(predictions)), dtype=np.float64)

                for j, pred in enumerate(predictions):
                    np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                    np.multiply(fant_ensemble_prediction, (1. / float(ens_size + 1)), out=fant_ensemble_prediction)
                    losses[j] = self.loss_function(labels, fant_ensemble_prediction)

            # -- Eval Iteration results
            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = rand.choice(all_best)  # break ties randomly
            ensemble_loss = losses[best]

            ensemble.append(predictions[best])
            trajectory.append(ensemble_loss)
            order.append(best)

            # Build Correct Validation loss list
            if not self.val_loss_over_iterations_:
                # Init
                self.val_loss_over_iterations_.append(ensemble_loss)
            elif self.val_loss_over_iterations_[-1] > ensemble_loss:
                # Improved
                self.val_loss_over_iterations_.append(ensemble_loss)
            else:
                # Not Improved
                self.val_loss_over_iterations_.append(self.val_loss_over_iterations_[-1])

            # -- Break for special cases
            #   - If we only have a pool of base models of size 1 (code found the single best model)
            #   - If we find a perfect ensemble/model, stop early
            if (len(predictions) == 1) or (ensemble_loss == 0):
                break

        self.indices_ = order
        self.trajectory_ = trajectory

    def _apply_use_best(self):
        # Last step of the algorithm as defined by  Caruana et al. 2004.
        #   (Basically from AutoGluon's code)
        min_score = np.min(self.trajectory_)
        idx_best = self.trajectory_.index(min_score)
        self.indices_ = self.indices_[:idx_best + 1]
        self.trajectory_ = self.trajectory_[:idx_best + 1]
        self.n_iterations = idx_best + 1
        self.validation_loss_ = self.trajectory_[idx_best]

    def _calculate_final_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=np.float64)

        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.n_iterations
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights


    # TODO: the MP implementation below was the most memory and time efficient implementation (on linux) that I found.
    #   This can certainly be improved in terms of readability but will likely decrease memory efficiency.
    # NOTE(eddiebergmane): Technically each process inherits everything when this is run, including all
    # the imports and their imports. One solution to speed it up even more is essentially to either have the computation function
    # in a standalone file. You could also look at `forkserver` where you can control what's preloaded with undocumented functionality
    # https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html
    # You could also try mapped memory with numpy loading
    # * https://stackoverflow.com/a/55545731/5332072
    def _compute_losses_mp(self, weighted_ensemble_prediction, labels, predictions, s):
        # -- Process Iteration Solutions
        func_args = (weighted_ensemble_prediction, labels, s, self.loss_function, predictions)
        pred_i_list = list(range(len(predictions)))

        with ProcessPoolExecutor(self._n_jobs, initializer=_pool_init, initargs=func_args) as ex:
            results = ex.map(_init_wrapper_evaluate_single_solution, pred_i_list)

        return np.array(list(results))


# -- Code for memory efficient MP
def _pool_init(_weighted_ensemble_prediction, _labels, _sample_size, _loss_function, _predictions):
    global p_weighted_ensemble_prediction, p_labels, p_sample_size, p_loss_function, p_predictions
    p_weighted_ensemble_prediction = _weighted_ensemble_prediction
    p_labels = _labels
    p_sample_size = _sample_size
    p_loss_function = _loss_function
    p_predictions = _predictions


def _init_wrapper_evaluate_single_solution(pred_index):
    return evaluate_single_solution(p_weighted_ensemble_prediction, p_labels, p_sample_size, p_loss_function,
                                    p_predictions[pred_index])


def evaluate_single_solution(weighted_ensemble_prediction, labels, sample_size, loss_function, pred):
    fant_ensemble_prediction = np.add(weighted_ensemble_prediction, pred)
    np.multiply(fant_ensemble_prediction, (1. / float(sample_size + 1)), out=fant_ensemble_prediction)

    return loss_function(labels, fant_ensemble_prediction)
