from __future__ import annotations

import numpy as np
import pandas as pd
from pytest_cases import parametrize

from byop.configspace import ConfigSpaceParser
from byop.pipeline import Pipeline, SpaceAdapter, choice, split, step
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

# Some toy data
X = pd.DataFrame({"a": ["1", "0", "1", "dog"], "b": [4, 5, 6, 7], "c": [7, 8, 9, 10]})
y = pd.Series([1, 0, 1, 1])


@parametrize("adapter", [ConfigSpaceParser()])
@parametrize("seed", range(10))
def test_split_with_choice(adapter: SpaceAdapter, seed: int) -> None:
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
            step("svm", SVC, space={"C": [0.1, 1, 10]}),
        ),
        name="test_pipeline_sklearn",
    )

    space = pipeline.space(parser=adapter)
    config = pipeline.sample(space, sampler=adapter, seed=seed)
    configured_pipeline = pipeline.configure(config)

    sklearn_pipeline = configured_pipeline.build()
    assert isinstance(sklearn_pipeline, SklearnPipeline)

    sklearn_pipeline = sklearn_pipeline.fit(X, y)
    sklearn_pipeline.predict(X)


@parametrize("adapter", [ConfigSpaceParser()])
@parametrize("seed", range(10))
def test_build_module(adapter: SpaceAdapter, seed: int) -> None:
    # Defining a pipeline
    pipeline = Pipeline.create(
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
            step("svm", SVC, space={"C": [0.1, 1, 10]}),
        ),
        name="test_pipeline_sklearn",
    )
    submodule_pipeline = pipeline.copy(name="sub")

    pipeline = pipeline.attach(modules=submodule_pipeline)

    space = pipeline.space(parser=adapter)

    config = pipeline.sample(space, sampler=adapter, seed=seed)

    configured_pipeline = pipeline.configure(config)

    # Build the pipeline and module
    built_pipeline = configured_pipeline.build()
    built_sub_pipeline = configured_pipeline.modules["sub"].build()

    assert isinstance(built_pipeline, SklearnPipeline)
    assert isinstance(built_sub_pipeline, SklearnPipeline)
