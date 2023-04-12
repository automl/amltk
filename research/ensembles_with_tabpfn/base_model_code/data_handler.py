import shutil
from pathlib import Path

import openml

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from byop.store import PathBucket

from research.ensembles_with_tabpfn.utils.config import FOLDS


def _obtain_data_sample(data_sample_name, X, y):
    fold_id, sample_id = [int(x[1:]) for x in data_sample_name.split("_")]

    # Do k-fold split
    train_index, test_index = list(StratifiedKFold(n_splits=len(FOLDS), shuffle=True,
                                                   random_state=270385).split(X, y))[fold_id]

    # Do re-sampling
    sample_random_state = int(f"347{fold_id}84")
    train_index, _ = train_test_split(train_index, random_state=sample_random_state, train_size=0.9,
                                      stratify=y[train_index], shuffle=True)

    # X_train, X_test, y_train, y_test
    X_train = X.iloc[train_index].reset_index(drop=True)
    X_test = X.iloc[test_index].reset_index(drop=True)
    return X_train, X_test, y[train_index], y[test_index]


def setup_data_bucket(openml_dataset_id: int, data_sample_name: str, seed: int, bucket_name: str) -> PathBucket:
    """Setup data bucket for experiment.
        -> Returns a bucket with "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"

        TODO: discuss if buckets are appropriate; for small datasets it is okay IMO
            but for larger datasets we would need load the data for each evaluation. Very expensive...

    Parameters
    ----------
    openml_dataset_id: int
        OpenML dataset ID to use for experiment.
    data_sample_name: str
        Defines which split of the data is returned.
    seed: int
        Used for generate random state for data split
    bucket_name: str
        Name of the bucket directory.
    """

    # TODO:
    #   - work on performing cross-validation for overall evaluation later
    #       (pass split as input or store splits in bucket?)
    #   - rework for final evaluation to avoid deleting previous results on accident
    #   - work on switching to tasks instead of dataset IDs such that we have a common / pre-defined validation protocol

    # -- Get data
    dataset = openml.datasets.get_dataset(openml_dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)


    # -- Blanket non-pipeline specific preprocessing
    # - Encode Classes as numbers
    y = LabelEncoder().fit_transform(y)
    # - Drop duplicated columns
    X = X.loc[:, ~X.columns.duplicated()].copy()
    # - Drop duplicated rows
    # X.drop_duplicates(inplace=True) # TODO decide on this

    # -- Split data
    X_train, X_test, y_train, y_test = _obtain_data_sample(data_sample_name, X, y)

    # -- Setup Data Bucket
    path = Path(bucket_name)

    # - Remove previous results if they exist
    if path.exists():
        shutil.rmtree(path)

    bucket = PathBucket(path)
    bucket.update(
        {
            "X_train.csv": X_train,
            "X_test.csv": X_test,
            "y_train.npy": y_train,
            "y_test.npy": y_test,
        }
    )

    return bucket
