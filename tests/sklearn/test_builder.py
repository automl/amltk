from __future__ import annotations

import numpy as np
import pandas as pd
from pytest_cases import parametrize
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from amltk.configspace import ConfigSpaceParser
from amltk.pipeline import Pipeline, SpaceAdapter, choice, group, split, step

# Some toy data
X = pd.DataFrame({"a": ["1", "0", "1", "dog"], "b": [4, 5, 6, 7], "c": [7, 8, 9, 10]})
y = pd.Series([1, 0, 1, 1])


def test_simple_pipeline() -> None:
    # Defining a pipeline
    pipeline = Pipeline.create(
        step("ordinal", OrdinalEncoder),
        step("std", StandardScaler),
        step("rf", RandomForestClassifier),
    )

    # Building the pipeline
    sklearn_pipeline: SklearnPipeline = pipeline.build()

    # Fitting the pipeline
    sklearn_pipeline.fit(X, y)

    # Predicting with the pipeline
    sklearn_pipeline.predict(X)


def test_passthrough() -> None:
    # Defining a pipeline
    step("passthrough", "passthrough")
    pipeline = Pipeline.create(
        step("passthrough", "passthrough"),
        split(
            "split",
            step("a", OrdinalEncoder),
            step("b", "passthrough"),
            item=ColumnTransformer,
            config={
                "a": make_column_selector(dtype_include=object),
                "b": make_column_selector(dtype_include=np.number),
            },
        ),
    )

    # Building the pipeline
    sklearn_pipeline: SklearnPipeline = pipeline.build()

    # Fitting the pipeline
    Xt = sklearn_pipeline.fit_transform(X, y)

    # Should ordinal encoder the strings
    assert np.array_equal(Xt[:, 0], np.array([1, 0, 1, 2]))

    # Should leave the remaining columns untouched
    assert np.array_equal(Xt[:, 1], np.array([4, 5, 6, 7]))
    assert np.array_equal(Xt[:, 2], np.array([7, 8, 9, 10]))


def test_simple_pipeline_with_group() -> None:
    # Defining a pipeline
    pipeline = Pipeline.create(
        group(
            "feature_preprocessing",
            step("ordinal", OrdinalEncoder) | step("std", StandardScaler),
        ),
        step("rf", RandomForestClassifier),
    )

    # Building the pipeline
    sklearn_pipeline: SklearnPipeline = pipeline.build()

    # Fitting the pipeline
    sklearn_pipeline.fit(X, y)

    # Predicting with the pipeline
    sklearn_pipeline.predict(X)


@parametrize("adapter", [ConfigSpaceParser()])
@parametrize("seed", range(10))
def test_split_with_choice(adapter: SpaceAdapter, seed: int) -> None:
    # Defining a pipeline
    pipeline = Pipeline.create(
        split(
            "feature_preprocessing",
            group(
                "categoricals",
                step("ordinal", OrdinalEncoder) | step("std", StandardScaler),
            ),
            group(
                "numericals",
                step("scaler", StandardScaler, space={"with_mean": [True, False]}),
            ),
            item=ColumnTransformer,
            config={
                "categoricals": make_column_selector(dtype_include=object),
                "numericals": make_column_selector(dtype_include=np.number),
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
    config = pipeline.sample(space=space, sampler=adapter, seed=seed)
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

    config = pipeline.sample(space=space, sampler=adapter, seed=seed)

    configured_pipeline = pipeline.configure(config)

    # Build the pipeline and module
    built_pipeline = configured_pipeline.build()
    built_sub_pipeline = configured_pipeline.modules["sub"].build()

    assert isinstance(built_pipeline, SklearnPipeline)
    assert isinstance(built_sub_pipeline, SklearnPipeline)
