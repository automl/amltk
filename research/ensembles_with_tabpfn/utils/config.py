from sklearn.metrics import balanced_accuracy_score
import numpy as np
from itertools import product


# -- Metrics to be pickleable
#   TODO: rework / overthink this (generator not pickleable...) but partial should work
def bacc_loss(y_true, y_pred):
    return 1 - balanced_accuracy_score(y_true, y_pred)


def bacc_loss_proba(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return 1 - balanced_accuracy_score(y_true, y_pred)


def bacc_proba(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)
    return balanced_accuracy_score(y_true, y_pred)


METRIC_MAP = {
    "balanced_accuracy": {
        "name": "balanced_accuracy",
        "function": balanced_accuracy_score,  # TODO: becomes more complicated for roc auc, maybe switch code
        "function_proba_in": bacc_proba,
        "requires_proba": False,
        "maximize": True,
        "loss_function": bacc_loss,
        "loss_function_proba_in": bacc_loss_proba,
        "to_loss_function": lambda x: 1 - x,
        "loss_to_score_function": lambda x:  1 - x,
        "task_type": "classification"
    }
}

# -- experiment configs
ALGO_NAMES = {"MLP", "RF", "LM", "GBM", "KNN", "XT"}
METRICS = set(METRIC_MAP.keys())
FOLDS = list(range(2))
SAMPLES = list(range(2))
DATASET_REF = {31}
C_MODEL = "XT"

ALL_EXPERIMENT_RUNS = list(product(ALGO_NAMES, METRICS, FOLDS, SAMPLES, DATASET_REF))
EXPERIMENT_RUNS_WO_ALGOS = list(product(METRICS, FOLDS, SAMPLES, DATASET_REF))