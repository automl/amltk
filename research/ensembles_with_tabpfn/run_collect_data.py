import numpy as np
import openml
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from byop.pipeline import Pipeline, choice, split, step


def _pipeline(algorithm_name):
    standard_preprocessing = split(
        "standard_preprocessing",
        [],
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numerics": make_column_selector(dtype_include=np.number),
        },
    )

    if algorithm_name in ["GLM", "MLP"]:
        print()

    pipeline = Pipeline.create(
        # Preprocessing
        split(
            "feature_preprocessing",
            (step("categoricals", SimpleImputer, space={
                "strategy": ["most_frequent", "constant"],
                "fill_value": ["missing"],
            })
             | step("ohe", OneHotEncoder,
                    space={
                        "min_frequency": (0.01, 0.1),
                        "handle_unknown": ["ignore", "infrequent_if_exist"],
                    },
                    config={"drop": "first"},
                    )
             ),
            (step("numerics", SimpleImputer, space={"strategy": ["mean", "median"]})
             | step("variance_threshold",
                    VarianceThreshold,
                    space={"threshold": (0.0, 0.2)}
                    )
             | choice("scaler",
                      step("standard", StandardScaler),
                      step("minmax", MinMaxScaler),
                      step("robust", RobustScaler),
                      step("passthrough", FunctionTransformer),
                      )
             ),
            item=ColumnTransformer,
            config={
                "categoricals": make_column_selector(dtype_include=object),
                "numerics": make_column_selector(dtype_include=np.number),
            },
        ),
        # Algorithm
        choice(
            "algorithm",
            step("svm", SVC, space={"C": (0.1, 10.0)}, config={"probability": True}),
            step("rf", RandomForestClassifier,
                 space={
                     "n_estimators": [10, 100],
                     "criterion": ["gini", "entropy", "log_loss"],
                 },
                 ),
            step("mlp", MLPClassifier,
                 space={
                     "activation": ["identity", "logistic", "relu"],
                     "alpha": (0.0001, 0.1),
                     "learning_rate": ["constant", "invscaling", "adaptive"],
                 },
                 ),
        ),
    )

    return


if __name__ == "__main__":
    print()
