import numpy as np

import math
from typing_extensions import Literal
from scipy.optimize import LinearConstraint

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    OrdinalEncoder,
)
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from byop.pipeline import Pipeline, choice, split, step, Step

from ConfigSpace import Constant, Float, Int, Categorical

ALGORITHM_NAMES = Literal["RF", "LM", "GBM", "lightgbm", "XGB", "CatBoost", "KNN", "XT", "MLP", "LR"]

def _get_preprocessing_step(algorithm_name: str) -> Step:
    """Builds preprocessing steps for an AutoML Toolkit pipeline.

    Parameters
    ----------
    algorithm_name: "RF", "LM", "GBM", "lightgbm", "XGB", "CatBoost", "KNN", "XT", "MLP", "LR"
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
    # TODO @LennartPurucker, extend this as you see fit
    if algorithm_name in ["MLP", "LR"]:
        preprocessing_step = preprocessing_step | choice(
            "scaler",
            step("standard", StandardScaler),
            step("minmax", MinMaxScaler),
            step("robust", RobustScaler),
            step("passthrough", FunctionTransformer)
        )

    return preprocessing_step


def _get_algorithm_step(algorithm_name: str) -> Step:
    """Builds an algorithm steps for an AutoML Toolkit pipeline.

    Parameters
    ----------
    algorithm_name: str
        Name of the algorithm for which the pipeline shall be created.
    """

    # TODO:
    #   - add metric agnostic algorithms! Right now: only for classification.
    #   - handle randomstate for each of theses algorithms!
    N_JOBS = -1

    if algorithm_name == "RF":
        # NOTE(eddiebergman): Reference:
        #
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/random_forest.py#L155

        # Technically AutoSklearn does X.shape[1] ** max_features (in [0.0, 1.0])
        # in the wrapped RandomForest itself. Sklearn only allows discrete number
        # of features or a fraction of features (X.shape[1] * max_features).
        # AutoSklearn puts emphasis when sampling towards taking a lower number of
        # features, so we emulate the same with taking a logspacing between (0.1, 1)
        # with 10 steps.
        _max_features = (np.logspace(0.1, 10, base=10, num=10) / 10).tolist()
        max_features = Categorical("max_features", _max_features, ordered=True)

        # TODO @LennartPurucker: `random_state` taken by "RF"
        algo_step = step(
            algorithm_name, RandomForestClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy"],
                "max_features": max_features,
                "min_samples_split": Int("min_samples_split", bounds=(2, 20), default=2),
                "min_samples_leaf": Int("min_samples_leaf", bounds=(1, 20), default=1),
                "bootstrap": Categorical("bootstrap", [True, False], default=True)

            },
            config = {
                "max_depth": None,
                "min_weight_fraction_leaf": 0.0,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "warm_start": True,
                "n_jobs": N_JOBS,
            }
        )
    elif algorithm_name == "LM":
        # NOTE(eddiebergman): Reference:
        #
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/sgd.py#L175
        #
        #   There's some additional conditionals that take place which I will leave out for now since we're not doing HPO
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/sgd.py#L230-L241

        # TODO @LennartPurucker: `random_state` taken by "LM"
        algo_step = step(
            algorithm_name, SGDClassifier,
            space={
                "penalty": Categorical("penalty", ["l2", "l1", "elasticnet"], default="l2"),
                "loss": Categorical("loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"], default="log"),
                "alpha": Float("alpha", bounds=(1e-7, 1e-1), default=1e-4),
                "l1_ratio": Float("l1_ratio", bounds=(1e-9, 1.0), default=0.15, log=True),
                "tol": Float("tol", bounds=(1e-5, 1e-1), default=1e-4, log=True),
                "epsilon": Float("epsilon", bounds=(1e-5, 1e-1), default=1e-4, log=True),
                "learning_rate": Categorical("learning_rate", ["constant", "optimal", "invscaling"], default="invscaling"),
                "eta0": Float("eta0", bounds=(1e-7, 1e-1), default=0.01, log=True),
                "power_t": Float("power_t", bounds=(1e-5, 1.0), default=0.5),
                "average": Categorical("average", [True, False], default=False),
            },
            config={
                "early_stopping": True,  # Adapted from autosklearn's iterative fit
                "max_iter": 1024,
                "fit_intercept": True,
                "shuffle": True,
                "warm_start": True,
                "n_jobs": N_JOBS,
            }
        )
    elif algorithm_name == "GBM":
        # NOTE(eddiebergman): Reference:
        #
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/gradient_boosting.py#L188
        #
        #   If doing HPO, then "early_stopping" is used to condition "n_iter_no_change", "validation_fraction".
        #   In default sklearn, the variable "n_iter_no_change" is used to determine  if early stopping happens but
        #   autosklearn introduces it's own variable on top of it.
        #
        #   For now we just used the defaults of GBC for early stopping
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/gradient_boosting.py#L241-L246

        # TODO @LennartPurucker: `random_state` taken in by "GBM"
        algo_step = step(
            algorithm_name, GradientBoostingClassifier,
            space={
                "n_estimators": [10, 100],
                "learning_rate": Float("learning_rate", bounds=(0.01, 1.0), default=0.1, log=True),
                "min_samples_leaf": Int("min_samples_leaf", bounds=(1, 200), default=20, log=True),
                "max_leaf_nodes": Int("max_leaf_nodes", bounds=(3, 2047), default=31, log=True),
                "criterion": ["friedman_mse", "squared_error"],
                "l2_regularization": Float("l2_regularization", bounds=(1e-10, 1.0), default=1e-10, log=True),
                "early_stopping": Categorical("early_stopping", [True, False], default=False),
            },
            config={
                "loss": "auto",
                "max_depth": None,
                "max_bins": 255,
                "tol": 1e-7,
                "scoring": "loss",
            }
        )

    elif algorithm_name == "lightgbm":
        # NOTE(eddiebergman): Reference
        #
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L1023
        #

        # TODO @LennartPurucker
        #   Seems LGMBClassfier also needs refrences to the `categorical_features`, `multiclass` and `objective`.
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L1045
        algo_step = step(
            algorithm_name, LGBMClassifier,
            space={
                "num_leaves": (5, 10),
                "max_depth": (3, 20),
                "learning_rate": Float("learning_rate", bounds=(-3, math.log(1)), log=True),
                "n_estimators": (50, 2000),
                "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                "subsample": (0.2, 0.8),
                "colsample_bytree": (0.2, 0.8),
                "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                "reg_lambda":[0, 1e-1, 1, 5, 10, 20, 50, 100],
            },
            config = {
                # "feature_fraction": 0.8,
                # "subsample": 0.2,
               "use_missing": True
                "n_jobs": N_JOBS
            }
        )

    elif algorithm_name == "XGB":
        # NOTE(eddiebergman): Reference
        #
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L1292
        #
        #   Search spaces seems weird but the paper has the following paragraph
        #
        #   > To be maximally fair to XGBoost, we also tried the search space of quadruple Kaggle grandmaster
        #   Bojan Tunguz (Tunguz, 2022), which we adapted slightly by using softmax instead of logistic, as we
        #   are in the multi-class setting. XGBoost with this search space performed worse for all considered
        #   time budgets than the search space by Shwartz-Ziv and Armon (2022).

        # TODO @LennartPurucker: `random_state` taken in by "XGB"
        algo_step = step(
            algorithm_name,
            XGBClassifier,
            space={
                "learning_rate": Float("learning_rate", bounds=(-7, math.log(1))),
                "max_depth": (1, 10),
                "subsample": (0.2, 1.0),
                "colsample_bytree": (0.2, 1.0),
                "colsample_bylevel": (0.2, 1.0),
                "min_child_weight": Float("min_child_weight" bounds=(-16, 5), log=True),
                "alpha": Float("alpha" bounds=(-16, 2), log=True),
                "lambda": Float("lambda" bounds=(-16, 2), log=True),
                "gamma": Float("gamme" bounds=(-16, 2), log=True),
                "n_estimators": (100, 4_000),
            },
            config = {
                "use_label_encoder": False,
                "nthread": N_JOBS,
            }
        )

    elif algorithm_name == "CatBoost":
        # NOTE(eddiebergman): Reference
        #
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L1231
        #
        #   Seems to also need `cat_features`, `random_seed` (which was set to sum(ys) in the paper). Also custom `loss_function`
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L88-L89

        # TODO @LennartPurucker: `random_seed` taken in by "CatBoost"
        algo_step = step(
            algorithm_name,
            CatBoostClassifier,
            space={
                "learning_rate": Float("learning_rate", bounds=(math.log(math.pow(math.e, -5)), math.log(1)), log=True),
                "random_strength": (1, 20),
                "l2_leaf_reg": Float("l2_leaf_reg", bounds=(math.log(1), math.log(10)), log=True),
                "bagging_temperature": (0.0, 0.1),
                "leaf_estimation_iterations": (1, 20),
                "iterations": (100, 4_000),
            },
            config = {
                "used_ram_limit": "4gb",
                "logging_level": "Silent",
                "thread_count": N_JOBS,
            }
        )

    elif algorithm_name == "KNN":
        # NOTE(eddiebergman): Reference
        #
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/k_nearest_neighbors.py#L63
        #
        #   AutoSklearn wraps in a OneVsRestClassifier is multiclass but I think we can skip this considering it's only really
        #   fit on X and just uses the majority label vote, not sure why OneVsRest was used.
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/k_nearest_neighbors.py#L29-L32
        algo_step = step(
            algorithm_name, KNeighborsClassifier,
            space={
                "n_neighbors": Int("n_neighbors", bounds=(1, 100), log=True, default=1),
                "weights": ["uniform", "distance"],
                "p": Categorical("p", [1, 2], default=2),
            },
            config = {
                "n_jobs": N_JOBS
            }
        )
    elif algorithm_name == "XT":
        # NOTE(eddiebergman): Reference:
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/extra_trees.py#L162

        # Technically AutoSklearn does X.shape[1] ** max_features (in [0.0, 1.0])
        # in the wrapped RandomForest itself. Sklearn only allows discrete number
        # of features or a fraction of features (X.shape[1] * max_features).
        # AutoSklearn puts emphasis when sampling towards taking a lower number of
        # features, so we emulate the same with taking a logspacing between (0.1, 1)
        # with 10 steps.

        # TODO @LennartPurucker: `random_state` taken for "MLP"
        _max_features = (np.logspace(0.1, 10, base=10, num=10) / 10).tolist()
        max_features = Categorical("max_features", _max_features, ordered=True)

        algo_step = step(
            algorithm_name, ExtraTreesClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy"],
                "min_samples_split": Int("min_samples_split", bounds=(2, 20), default=2),
                "min_samples_leaf": Int("min_samples_leaf", bounds=(1, 20), default=1),
                "bootstrap": [True, False],
            },
            config = {
                "max_depth": None,
                "min_weight_fraction_leaf": 0.0,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "warm_start": True,
                "n_jobs": N_JOBS
            }

        )
    elif algorithm_name == "MLP":
        # NOTE(eddiebergman): Reference
        #
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/autosklearn/pipeline/components/classification/mlp.py#L209
        #

        # TODO @LennartPurucker: `random_state` taken for "MLP""
        algo_step = step(
            algorithm_name, MLPClassifier,
            space={
                "hidden_layer_depth": Int("hidden_layer_depth", bounds=(1, 3), default=1),
                "num_nodes_per_layer": Int("num_nodes_per_layer", bounds=(16, 264), log=True, default=32),
                "activation": Categorical("activation", ["tanh", "relu"], default="relu"),
                "alpha": Float("alpha", bounds=(1e-7, 1e-1), default=1e-4, log=True),
                "learning_rate_init": Categorical("learning_rate_init", bounds=(1e-4, 0.5), default=1.3, log=True),
                "early_stopping": Categorical("early_stopping", [True, False], default=True),
                "learning_rate": ["constant", "invscaling", "adaptive"],
            },
            # Constants set by asklearn
            config = {
                "n_iter_no_change": 32,
                "validation_fraction": 0.1,
                "tol": 1e-4,
                "solver": "adam",
                "batch_size": "auto",
                "shuffle": True,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-8,
                "warm_start": True,
            }
        )
    elif algorithm_name == "LR": # LogisticRegression
        # NOTE(eddiebergman): Reference
        #
        #   Space reference:
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L1050
        #
        #   AutoSklearn uses LibLinear instead as athe linear model, went with TabPFN code version:
        #   * https://github.com/search?q=repo%3Aautoml%2Fauto-sklearn%20LogisticRegression&type=code
        #   * https://github.com/automl/auto-sklearn/blob/673211252ca508b6f5bb92cf5fa87c6455bbad99/misc/classifiers.csv#L16
        #
        #   They also apply preprocessing specifically like this:
        #   * https://github.com/automl/TabPFN/blob/889fad7070ded19ac3b247daf47d94b2538695cb/tabpfn/scripts/tabular_baselines.py#L208-L238
        #
        #   I added to the list of algorithms that get's standardization above.
        #   In the TabPFN code, they specifically only use a MinMaxScaler but I think it's fine how we have it now.

        # TODO @LennartPurucker: `random_state` taken for "LR".
        algo_step = step(
            algorithm_name, LogisticRegression,
            space={
                "penalty": ["l2", "l1", "none"],  # "none" was deprecated in sklearn 1.2 and removed in 1.4 for None
                                                  # ConfigSpace can't handle None in it's categoricals :'(
                "max_iter": (50, 500),
                "fit_intercept": [True, False],
                "C": Float("C", bounds=(-5, math.log(5.0)), log=True)
            },
            config = {
                "solver": "saga",
                "tol": 1e-4,
                "n_jobs": N_JOBS,
            }
        )
    else:
        raise ValueError(f"Algorithm {algorithm_name} not supported!")

    return algo_step


def build_pipeline(algorithm_name: ALGORITHM_NAMES) -> Pipeline:
    """Builds a AutoML Toolkit pipeline for a given algorithm.

    Parameters
    ----------
    algorithm_name: "RF", "LM", "GBM", "lightgbm", "XGB", "CatBoost", "KNN", "XT", "MLP", "LR"
        Name of the algorithm for which the pipeline shall be created.

    Returns
    -------
    Pipeline
        An AutoML Toolkit pipeline for the given algorithm.
    """

    preprocessing_step = _get_preprocessing_step(algorithm_name)
    algorithm_step = _get_algorithm_step(algorithm_name)

    return Pipeline.create(preprocessing_step | algorithm_step)
