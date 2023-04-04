"""
Place to put code related to preprocessing for (post hoc) ensembling like
    * preselection/pruning of the pool of base models
    * ?probability calibration?
    * ?threshold tuning?
"""

from typing import List, Dict, Tuple
import math


# --- Pruning
def _prune_to_top_n(base_models, n):
    """Prune to the top N models based on validation score."""
    maximize_metric = True  # TODO: get this from the metric / outer scope / scorer
    base_models = sorted(base_models, key=lambda m: m.val_score, reverse=maximize_metric)

    return base_models[:n]


def _prune_to_silo_top_n(base_models, n):
    """Prune to the top N models per silo based on validation score. A silo represents an algorithm family."""
    # Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)

    maximize_metric = True  # TODO: get this from the metric / outer scope / scorer

    # -- Get silos
    # algorithm family (af) to list of base models of this family
    af_to_model = {}  # type: Dict[str, List[Tuple[float, str, str]]]
    for bm in base_models:
        algorithm_family = bm.config["algorithm"]
        if algorithm_family not in af_to_model:
            af_to_model[algorithm_family] = []

        af_to_model[algorithm_family].append((bm.val_score, bm.name, algorithm_family))

    # Get the minimal number of entire a silo can have (in case of equal distribution)
    min_silo_val = max(math.floor(n / len(af_to_model.keys())), 1)

    while sum(len(base_models) for base_models in af_to_model.values()) > n:
        # Find silos with too many values
        too_large_silos = [af for af, base_models in af_to_model.items() if len(base_models) > min_silo_val]

        if not too_large_silos:
            break

        # For all silos with too many values, find and remove the model with the worst performance
        #   - This won't remove silos entirely, because silos with at least 1 element won't be too large
        #   - The first element of sorted() is the element with the worst performance (highest/lowest value)
        worst_model = sorted(
            [base_model for af in too_large_silos for base_model in af_to_model[af]],
            key=lambda x: x[0],  # sort by validation score/loss
            reverse=maximize_metric  # determines if higher or lower is better
        )[0]  # select the worst model (first element)
        af_to_model[worst_model[-1]].remove(worst_model)

    if sum(len(base_models) for base_models in af_to_model.values()) > n:
        # In this case, we have more silos than top_n (cat_to_model.keys() > top_n)
        # Moreover, at this point, all silos will only have 1 element in it.
        # To resolve this, we can simply return the top_n models over these silos
        # (other fallbacks like random for more diversity would work as well, but we think top is best for now)
        sort_rest = sorted(
            [base_model for base_models in af_to_model.values() for base_model in base_models],  # all models
            key=lambda x: x[0], reverse=maximize_metric
        )
        bm_names_to_keep = [name for _, name, _ in sort_rest[-n:]]
    else:
        bm_names_to_keep = [name for vals in af_to_model.values() for _, name, _ in vals]

    base_models = [base_model for base_model in base_models if base_model.name in bm_names_to_keep]

    # Sort to have similar order to top n (also can be beneficial for ensembling)
    return sorted(base_models, key=lambda m: m.val_score, reverse=maximize_metric)


def prune_base_models(base_models: List[object], max_number_base_models: int = 25, pruning_method: str = "TopN"):
    """Prunes the base models to a maximum number of base models.

    Parameters
    ----------
    base_models: List[base_model_object]
        TODO define base model object to be used here or whatever we decide on later
        Base model object that includes validation score, validation predictions, and model config.
    max_number_base_models: int
        The final number of base models (at most).
    pruning_method: str in {"SiloTopN", "TopN"}, default="TopN"
        The method used to prune the base models.
            * TopN: Prune to the top N models based on validation score.
            * SiloTopN: Pruned to N, such that as many top-performing models of each algorithm family are kept.
            * X: more method possible... would be cool to research this (with the AutoML toolkit)...
    """

    if pruning_method == "SiloTopN":
        base_models = _prune_to_silo_top_n(base_models, max_number_base_models)
    elif pruning_method == "TopN":
        base_models = _prune_to_top_n(base_models, max_number_base_models)
    else:
        raise ValueError(f"Pruning method {pruning_method} not supported.")

    return base_models
