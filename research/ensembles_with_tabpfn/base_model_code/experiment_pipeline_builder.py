import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    OrdinalEncoder,
)

from byop.pipeline import Pipeline, choice, split, step


def _get_preprocessing_step(algorithm_name: str) -> step:
    """Builds preprocessing steps for an AutoML Toolkit pipeline.

    Parameters
    ----------
    algorithm_name: str
        Name of the algorithm for which the pipeline shall be created.
    """

    # -- Standard Preprocessing
    nan_handling_and_cat_encoding = split(
        "nan_handling_and_cat_encoding",
        # Categorical Data
        (step("categoricals", SimpleImputer,
              space={"strategy": ["most_frequent", "constant"], "fill_value": ["missing"]})
         | choice("cat_encoding",
                  # Encoding Options
                  step(
                      "ohe", OneHotEncoder,
                      space={"min_frequency": (0.01, 0.1), "handle_unknown": ["ignore", "infrequent_if_exist"]},
                      config={"drop": "first"}
                  ),
                  step("oe", OrdinalEncoder)
                  )
         ),
        # Numerical Data
        (step("numerics", SimpleImputer, space={"strategy": ["mean", "median"]})),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    )

    # -- default numerical preprocessing
    feature_cleaner = step("variance_threshold", VarianceThreshold, config={"threshold": 0.0})

    # -- Build standard preprocessing
    preprocessing_step = nan_handling_and_cat_encoding | feature_cleaner

    # -- Add algorithm specific preprocessing
    if algorithm_name in ["GLM", "MLP"]:
        preprocessing_step = preprocessing_step | choice(
            "scaler",
            step("standard", StandardScaler),
            step("minmax", MinMaxScaler),
            step("robust", RobustScaler),
            step("passthrough", FunctionTransformer)
        )

    return preprocessing_step


def _get_algorithm_step(algorithm_name: str) -> step:
    """Builds an algorithm steps for an AutoML Toolkit pipeline.

    Parameters
    ----------
    algorithm_name: str
        Name of the algorithm for which the pipeline shall be created.
    """

    # TODO:
    #   - add search space for each algorithm correctly! Right now: just basic defaults for testing purposes
    #   - add metric agnostic algorithms! Right now: only for classification.
    #   - handle n_jobs for each algorithm; we should use -1 and parallelize on an algorithm level
    #   - handle randomstate for each of theses algorithms!

    if algorithm_name == "RF":
        algo_step = step(
            algorithm_name, RandomForestClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy", "log_loss"],
            }
        )
    elif algorithm_name == "LM":
        algo_step = step(
            algorithm_name, SGDClassifier,
            space={
                "penalty": ["l2", "l1", "elasticnet"],
            },
            config={
                "early_stopping": True,
                "max_iter": 2000,
                "n_iter_no_change": 5,
                "loss": "log_loss"
            }
        )
    elif algorithm_name == "GBM":
        algo_step = step(
            algorithm_name, GradientBoostingClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["friedman_mse", "squared_error"],
            },
            config={
                "n_iter_no_change": 5
            }
        )
    elif algorithm_name == "KNN":
        algo_step = step(
            algorithm_name, KNeighborsClassifier,
            space={
                "n_neighbors": [5, 15, 30],
                "weights": ["uniform", "distance"],
            }
        )
    elif algorithm_name == "XT":
        algo_step = step(
            algorithm_name, ExtraTreesClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy", "log_loss"],
            }
        )
    elif algorithm_name == "MLP":
        algo_step = step(
            algorithm_name, MLPClassifier,
            space={
                "activation": ["identity", "logistic", "relu"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            }
        )
    else:
        raise ValueError(f"Algorithm {algorithm_name} not supported!")

    return algo_step


def build_pipeline(algorithm_name: str) -> Pipeline:
    """Builds a AutoML Toolkit pipeline for a given algorithm.

    Parameters
    ----------
    algorithm_name: str in {"MLP", "RF", "LM", "GBM", "KNN", "XT"}
        Name of the algorithm for which the pipeline shall be created.

    Returns
    -------
    Pipeline
        An AutoML Toolkit pipeline for the given algorithm.
    """

    preprocessing_step = _get_preprocessing_step(algorithm_name)
    algorithm_step = _get_algorithm_step(algorithm_name)

    return Pipeline.create(preprocessing_step | algorithm_step)
