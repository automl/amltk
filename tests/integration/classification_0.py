from __future__ import annotations

from typing import TypeVar

import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from byop.configspace import ConfigSpacePipeline, choice, step

Self = TypeVar("Self", bound="Classifier")


class NaNRemover:
    def __init__(self, axis: int = 1):
        self.axis = axis

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X[~np.isnan(X).any(axis=self.axis)]


class Classifier:
    pipeline = ConfigSpacePipeline(
        choice(
            "preprocessing",
            step("NanRemover", NaNRemover),
            step(
                "imputer",
                SimpleImputer,
                space=ConfigurationSpace(
                    {"strategy": ["mean", "median", "most_frequent"]}
                ),
            ),
        ),
        choice(
            "scaler",
            step("standard", StandardScaler),
            step("min-max", MinMaxScaler),
        ),
        step(
            "classifier",
            DummyClassifier,
            space=ConfigurationSpace(
                {"strategy": ["most_frequent", "prior", "stratified", "uniform"]}
            ),
            inject=["seed"],
        ),
    )

    def __init__(
        self,
        preprocessing: SimpleImputer | NaNRemover,
        scaler: StandardScaler | MinMaxScaler,
        classifier: DummyClassifier,
    ):
        self.preprocessing = preprocessing
        self.scaler = scaler
        self.classifier = classifier

    def fit(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        X = self.preprocessing.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.preprocessing.transform(X)
        X = self.scaler.transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.preprocessing.transform(X)
        X = self.scaler.transform(X)
        return self.classifier.predict_proba(X)

    @classmethod
    def sample(cls: type[Self], seed: int | None = None) -> Self:
        config = cls.pipeline.space(seed=seed).sample_configuration()
        pipeline = cls.pipeline.build(config, seed=seed)
        return cls(**pipeline)

    def __repr__(self) -> str:
        return f"Classifier({self.preprocessing}, {self.scaler}, {self.classifier})"


if __name__ == "__main__":
    XX, yy = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=4,
        n_classes=4,
    )
    XX.ravel()[np.random.choice(XX.size, 15, replace=False)] = np.nan
    pipeline = Classifier.sample()

    pipeline.fit(XX, yy)
    pipeline.predict(XX)
    print(pipeline)
