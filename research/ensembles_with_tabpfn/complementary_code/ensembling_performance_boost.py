from byop.ensembling.ensemble_weighting import GreedyEnsembleSelection, EnsembleWeightingCMAES
from typing import List, Callable
from research.ensembles_with_tabpfn.complementary_code.data_handler import \
    FakedFittedAndValidatedClassificationBaseModel as BaseModel


def _get_ensemble_method_predictions(base_models: List[BaseModel], n_iterations, loss_function_proba_in, n_jobs, seed,
                                     X_val, y_val, X_test):

    # Switch to val simulation
    for bm in base_models:
        bm.switch_to_val_simulation()

    ens_base = EnsembleWeightingCMAES  # EnsembleWeightingCMAES, GreedyEnsembleSelection
    # TODO: make metric agnostic (would crash for proba metric right now)
    ens = ens_base(base_models, n_metric_evals=1000, loss_function=loss_function_proba_in,
                   n_jobs=n_jobs, random_state=seed)
    ens.fit(X_val, y_val)
    # print("Final Weights: ", ens.weights_, " for base models: ", len(base_models))
    # Switch to predict simulation
    for bm in base_models:
        bm.switch_to_test_simulation()

    return ens.predict_proba(X_test)


def get_data_for_performance_increase_with_new_model(base_models: List[BaseModel], complement_model: BaseModel,
                                        loss_function_proba_in: Callable, score_function_proba_in: Callable, seed: int,
                                        X_val, X_test, y_val, y_test, n_iterations=50, n_jobs=1):
    """Obtain the performance increase when the complement model is added to the base models.

    Parameters
    ----------
    base_models: List[BaseModel]
        The default base models that are to be used for the ensemble.
    complement_model
        The model that is to be analysed w.r.t. its complementarity.
    loss_function_proba_in
        A function that takes [y_true, y_pred_proba] and returns a loss value. Used by the ensemble.
    score_function_proba_in:
        A function that takes [y_true, y_pred_proba] and returns a score value. Used to return the final score.
    seed: int
        used by ensemble method to create random state.
    n_iterations: int, default=50
        Number of iterations, results in len(base_models) * n_iterations metric function evaluations.
    n_jobs: int, default=-1
        Algorithm-level multiprocessing. -1 means using all available cores.

    Returns
    -------

    """

    bm_predictions = _get_ensemble_method_predictions(base_models, n_iterations, loss_function_proba_in, n_jobs, seed,
                                                      X_val, y_val, X_test)
    bm_with_complement_predictions = _get_ensemble_method_predictions(base_models + [complement_model], n_iterations,
                                                                      loss_function_proba_in, n_jobs, seed,
                                                                      X_val, y_val, X_test)

    score_bm = score_function_proba_in(y_test, bm_predictions)
    score_bm_with_complement = score_function_proba_in(y_test, bm_with_complement_predictions)

    return bm_predictions, bm_with_complement_predictions, score_bm, score_bm_with_complement
