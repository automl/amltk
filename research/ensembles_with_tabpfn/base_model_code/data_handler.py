import shutil
from pathlib import Path

import openml

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from byop.store import PathBucket


def setup_data_bucket(seed: int, bucket_name:str, openml_dataset_id: int = 31) -> PathBucket:
    """Setup data bucket for experiment.
        -> Returns a bucket with "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"

        TODO: discuss if buckets are appropriate; for small datasets it is okay IMO
            but for larger datasets we would need load the data for each evaluation. Very expensive...

    Parameters
    ----------
    seed: int
        Used for generate random state for data split
    bucket_name: str
        Name of the bucket directory.
    openml_dataset_id: int
        OpenML dataset ID to use for experiment.
    """

    # TODO:
    #   - work on performing cross-validation for overall evaluation later
    #       (pass split as input or store splits in bucket?)
    #   - work on storing results in a persistent way across runs for different algorithms/datasets/splits
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.2, stratify=y
    )

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
