# Parse data for this part of our experiments from base model's data

from dataclasses import dataclass
import re
from typing import List, Tuple

from byop.store import PathBucket

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import LabelEncoder


@dataclass
class FakedFittedAndValidatedClassificationBaseModel:
    """Fake sklearn-like base model (classifier) usable by ensembles in the same way as real base models.

    To simulate validation and test predictions, we start by default with returning validation predictions.
    Then, after fitting the ensemble on the validation predictions, we switch to returning test predictions using
        `switch_to_test_simulation`.
    """

    name: str
    config: dict
    val_probabilities: List[np.ndarray]
    val_score: float
    test_probabilities: List[np.ndarray]
    le_: LabelEncoder
    classes_: np.ndarray
    return_val_data: bool = True

    @property
    def probabilities(self):
        if self.return_val_data:
            return self.val_probabilities
        return self.test_probabilities

    def predict(self, X):
        return np.argmax(self.probabilities, axis=1)

    def predict_proba(self, X):
        return self.probabilities

    def switch_to_test_simulation(self):
        self.return_val_data = False

    def switch_to_val_simulation(self):
        self.return_val_data = True


def read_all_base_models(path_to_base_model_data: str, dataset_ref: str, data_sample_name: str, algorithms: List[str]) \
        -> Tuple[List[FakedFittedAndValidatedClassificationBaseModel], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read all base models as fake base model for our usage later on."""
    X_train, X_test, y_train, y_test = None, None, None, None
    le_, classes_ = None, None
    base_models = []

    for algo_name in algorithms:
        bm_data_bucket = PathBucket(path_to_base_model_data + f"/{algo_name}/{dataset_ref}/{data_sample_name}")

        # -- Obtain all base model paths for this algorithm
        pattern = r"trial_(?P<trial>.+)_val_probabilities.npy"
        trial_names = [
            match.group("trial")
            for key in bm_data_bucket
            if (match := re.match(pattern, key)) is not None
        ]

        tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test = (
            bm_data_bucket["X_train.csv"].load(),
            bm_data_bucket["X_test.csv"].load(),
            bm_data_bucket["y_train.npy"].load(),
            bm_data_bucket["y_test.npy"].load(),
        )

        if X_train is None:
            X_train, X_test, y_train, y_test = tmp_X_train, tmp_X_test, tmp_y_train, tmp_y_test

            # Store the classes seen during fit of base models (ask Lennart why, very specific edge case reason...)
            le_ = LabelEncoder().fit(y_train)
            classes_ = le_.classes_
        else:
            # Sanity Check: original data must be equal for all algorithms!
            pd.testing.assert_frame_equal(X_train, tmp_X_train)
            pd.testing.assert_frame_equal(X_test, tmp_X_test)
            assert_array_equal(y_train, tmp_y_train)
            assert_array_equal(y_test, tmp_y_test)

        # -- Obtain all base model data for this algorithm
        val_score_key = [k for k in bm_data_bucket[f"trial_{trial_names[0]}_scores.json"].load().keys()
                         if k.startswith("validation_")][0]
        tmp_base_models = [
            FakedFittedAndValidatedClassificationBaseModel(
                f"{algo_name}_" + name,  # avoid duplicate names; TODO: maybe move this to name creation/save...
                bm_data_bucket[f"trial_{name}_config.json"].load(),
                bm_data_bucket[f"trial_{name}_val_probabilities.npy"].load(check=np.ndarray),
                bm_data_bucket[f"trial_{name}_scores.json"].load()[val_score_key],
                bm_data_bucket[f"trial_{name}_test_probabilities.npy"].load(check=np.ndarray),
                le_,
                classes_,
            )
            for name in trial_names
        ]

        # - Add algorithm information to config because it is missing
        # TODO: maybe move this to config creation/save...
        for bm in tmp_base_models:
            bm.config["algorithm"] = algo_name

        base_models.extend(tmp_base_models)

    return base_models, X_train, X_test, y_train, y_test