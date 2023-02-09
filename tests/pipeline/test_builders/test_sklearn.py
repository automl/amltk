import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from byop import Pipeline, choice, split, step

# Some toy data
X = pd.DataFrame({"a": ["1", "0", "1", "dog"], "b": [4, 5, 6, 7], "c": [7, 8, 9, 10]})
y = pd.Series([1, 0, 1, 1])


def test_split_with_choice():
    # Defining a pipeline
    pipeline = Pipeline.create(
        split(
            "feature_preprocessing",
            step("cats", OrdinalEncoder) | step("std", StandardScaler),
            step("nums", StandardScaler),
            item=ColumnTransformer,
            config={
                "cats": make_column_selector(dtype_include=object),
                "nums": make_column_selector(dtype_include=np.number),
            },
        ),
        step(
            "another_standard_scaler",
            StandardScaler,
            config={"with_mean": False},
        ),
        choice(
            "algorithm",
            step(
                "rf",
                item=RandomForestClassifier,
                space={
                    "n_estimators": [10, 100],
                    "criterion": ["gini", "entropy", "log_loss"],
                },
            ),
            step("svm", SVC, config={"C": [0.1, 1, 10]}),
        ),
        name="test_pipeline_sklearn",
    )

    space = pipeline.space(seed=1)
    config = space.sample_configuration()
    configured_pipeline = pipeline.configure(config)

    sklearn_pipeline = configured_pipeline.build()
    assert isinstance(sklearn_pipeline, SklearnPipeline)

    sklearn_pipeline = sklearn_pipeline.fit(X, y)
    sklearn_pipeline.predict(X)
