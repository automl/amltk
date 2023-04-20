import numpy as np

from itertools import product
from functools import partial

from research.ensembles_with_tabpfn.utils.custom_metrics.roc_auc import roc_auc_score
from sklearn.metrics import balanced_accuracy_score


# -- Metrics to be pickleable
#   TODO: rework / overthink this (generator not pickleable...) but partial should work

# - Balanced Accuracy
def bacc_loss(y_true, y_pred):
    return 1 - balanced_accuracy_score(y_true, y_pred)


def bacc_loss_proba(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return 1 - balanced_accuracy_score(y_true, y_pred)


def bacc_proba(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return balanced_accuracy_score(y_true, y_pred)


# - ROC AUC
def roc_auc_labels(y_true, y_pred):
    raise ValueError("ROC AUC requires prediction probabilities!")


def roc_auc_binary_proba(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba[:, 1])


def roc_auc_binary_loss_proba(y_true, y_pred_proba):
    return 1 - roc_auc_score(y_true, y_pred_proba[:, 1])


def roc_auc_multi_proba(y_true, y_pred_proba, labels):
    return roc_auc_score(y_true, y_pred_proba, average="macro", multi_class="ovr", labels=labels)


def roc_auc_multi_loss_proba(y_true, y_pred_proba, labels):
    return 1 - roc_auc_score(y_true, y_pred_proba, average="macro", multi_class="ovr", labels=labels)


def _init_roc_auc_multi_proba(is_binary, labels):
    tmp = METRIC_MAP["roc_auc"].copy()
    if is_binary:
        tmp["function_proba_in"] = roc_auc_binary_proba
        tmp["loss_function_proba_in"] = roc_auc_binary_loss_proba
    else:
        tmp["function_proba_in"] = partial(roc_auc_multi_proba, labels=labels)
        tmp["loss_function_proba_in"] = partial(roc_auc_multi_loss_proba, labels=labels)

    return tmp


METRIC_MAP = {
    "balanced_accuracy": {
        "name": "balanced_accuracy",
        "function": balanced_accuracy_score,
        "function_proba_in": bacc_proba,
        "requires_proba": False,
        "maximize": True,
        "loss_function": bacc_loss,
        "loss_function_proba_in": bacc_loss_proba,
        "to_loss_function": lambda x: 1 - x,
        "loss_to_score_function": lambda x: 1 - x,
        "task_type": "classification",
        "requires_init": False
    },
    "roc_auc": {
        "name": "roc_auc",
        "function": roc_auc_labels,
        "function_proba_in": None,
        "requires_proba": True,
        "maximize": True,
        "loss_function": roc_auc_labels,
        "loss_function_proba_in": None,
        "to_loss_function": lambda x: 1 - x,
        "loss_to_score_function": lambda x: 1 - x,
        "task_type": "classification",
        "requires_init": True,
        "init_func": _init_roc_auc_multi_proba
    }
}

def init_metric_data(metric_data, meta_data):
    if metric_data["requires_init"]:
        n_classes = meta_data["n_classes"]
        metric_data = metric_data["init_func"](n_classes == 2, list(range(n_classes)))

    return metric_data

# -- experiment configs
# TODO @LennartPurucker: Probably need to update the AlgoNames
# ALGORITHM_NAMES = Literal["RF", "LM", "GBM", "lightgbm", "XGB", "CatBoost", "KNN", "XT", "MLP", "LR"]
ALGO_NAMES = {"MLP", "RF", "LM", "GBM", "KNN", "XT"}
METRICS = set(METRIC_MAP.keys())  #  ["balanced_accuracy"]
FOLDS = list(range(10))
SAMPLES = list(range(10))
DATASET_REF = {31}
C_ALGO = "LM"

# TODO: think about making HPs its own experiment hyperparameter and parallelization level

ALL_EXPERIMENT_RUNS = list(product(ALGO_NAMES, METRICS, FOLDS, SAMPLES, DATASET_REF))
EXPERIMENT_RUNS_WO_ALGOS = list(product(METRICS, FOLDS, SAMPLES, DATASET_REF))
EXPERIMENT_HIGH_LEVEL = list(product(METRICS, DATASET_REF))
