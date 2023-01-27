import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from byop import Pipeline, choice, split, step
from byop.sklearn.builder import build

# Some toy data
X = pd.DataFrame({"a": ["1", "0", "1", "dog"], "b": [4, 5, 6, 7], "c": [7, 8, 9, 10]})
y = pd.Series([1, 0, 1, 1])


def test_split_with_choice():

    # Defining a pipeline
    pipeline = Pipeline.create(
        split(
            "feature_preprocessing",
            step("cats", OrdinalEncoder) | step("scalerize", OneHotEncoder),
            step("nums", StandardScaler),
            item=ColumnTransformer,
            config={
                "cats": make_column_selector(dtype_include=object),
                "nums": make_column_selector(dtype_include=np.number),
            },
        ),
        choice(
            "algorithm",
            step("rf", RandomForestClassifier, space={"n_estimators": [10, 100]}),
            step("svm", SVC, space={"C": [0.1, 1, 10]}),
        ),
    )

    space = pipeline.space()

    config = space.sample_configuration()

    configured_pipeline = pipeline.configure(config)

    sklearn_pipeline = build(configured_pipeline)

    sklearn_pipeline = sklearn_pipeline.fit(X, y)
    sklearn_pipeline.predict(X)
