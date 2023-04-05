from __future__ import annotations

import operator
import shutil
from pathlib import Path
from typing import Callable, Iterator, Literal, TypeVar

import numpy as np
import pandas as pd
import pytest
from pytest_cases import case, fixture, parametrize, parametrize_with_cases

from byop.store import Bucket, PathBucket

T = TypeVar("T")
DF = TypeVar("DF", pd.DataFrame, pd.Series)


def xfail(thing: object, reason: str):
    return pytest.param(thing, marks=pytest.mark.xfail(reason=reason))


def unsupported_format(thing: object):
    return xfail(thing, "Unsupported format: https://github.com/automl/byop/issues/4")


@case
def bucket_path_bucket(tmp_path: Path) -> Iterator[PathBucket]:
    path = tmp_path / "bucket"
    yield PathBucket(path)
    shutil.rmtree(path)


def unjson_serialisable(x):
    return x


@fixture
@parametrize_with_cases("bucket", cases=".", prefix="bucket_")
def bucket(bucket: Bucket) -> Bucket:
    return bucket


@parametrize(extension=("npy", unsupported_format("npz")))
def data_numpy_array_npy(
    extension: str,
) -> tuple[
    np.ndarray,
    str,
    type[np.ndarray],
    Callable[[np.ndarray, np.ndarray], bool],
]:
    array = np.random.rand(10, 10)
    return array, f"array.{extension}", np.ndarray, np.array_equal


@parametrize(
    extension=(
        "csv",
        unsupported_format("feather"),
        unsupported_format("h5"),
        unsupported_format("hdf"),
        unsupported_format("hdf5"),
        "parquet",
        unsupported_format("xls"),
        unsupported_format("xlsx"),
    )
)
@parametrize(kind=(unsupported_format("series"), "frame"))
@parametrize(index=("named", "unnamed"))
def data_pandas(
    extension: str,
    index: str,
    kind: Literal["series", "frame"],
) -> tuple[DF, str, type[DF], Callable[[DF, DF], bool]]:
    if kind == "series" and extension in ("csv"):
        pytest.skip("Series not supported for this extension")

    if kind == "series":
        df = pd.Series(np.random.randint(0, 10, size=3), name="ABC")
        check = pd.Series
    else:
        df = pd.DataFrame(np.random.randint(0, 10, size=(3, 3)), columns=list("ABC"))
        check = pd.DataFrame

    if index == "named":
        df.index.name = "index"
    else:
        df.index.name = None
    return df, f"df.{extension}", check, pd.DataFrame.equals  # type: ignore


def data_string() -> tuple[str, str, type[str], Callable[[str, str], bool]]:
    return "Hello World", "string.txt", str, operator.eq


def data_bytes() -> tuple[bytes, str, type[bytes], Callable[[bytes, bytes], bool]]:
    pytest.xfail(
        "bytes adds some werid prepeding" " see https://github.com/automl/byop/issues/4"
    )
    return b"Hello World", "bytes.bin", bytes, operator.eq


def data_dict_json() -> tuple[dict, str, type[dict], Callable[[dict, dict], bool]]:
    return {"a": 1, "b": 2}, "dict.json", dict, operator.eq


def data_dict_pickle() -> tuple[dict, str, type[dict], Callable[[dict, dict], bool]]:
    return {"b": unjson_serialisable}, "dict.pkl", dict, operator.eq


def data_dict_yaml() -> tuple[dict, str, type[dict], Callable[[dict, dict], bool]]:
    return {"a": 1, "b": 2}, "dict.yaml", dict, operator.eq


def data_list_json() -> tuple[list, str, type[list], Callable[[list, list], bool]]:
    return [1, 2, 3], "list.json", list, operator.eq


def data_list_pickle() -> tuple[list, str, type[list], Callable[[list, list], bool]]:
    return [unjson_serialisable], "list.pkl", list, operator.eq


def data_list_yaml() -> tuple[list, str, type[list], Callable[[list, list], bool]]:
    return [1, 2, 3], "list.yaml", list, operator.eq


def data_pickle() -> tuple[int, str, type[int], Callable[[int, int], bool]]:
    return 42, "pickle.pkl", int, operator.eq


@parametrize_with_cases("item, key, check, equal", cases=".", prefix="data_")
def test_bucket(
    bucket: PathBucket,
    item: T,
    key: str,
    check: type[T],
    equal: Callable[[T, T], bool],
) -> None:
    bucket[key] = item
    assert bucket[key].exists()
    assert key in bucket
    assert len(bucket) == 1

    retrieved = bucket[key].load()
    assert equal(item, retrieved)

    retrieved = bucket[key].get()
    assert equal(item, retrieved)  # type: ignore

    retrieved = bucket[key].get(check=check)
    assert equal(item, retrieved)

    bucket[key].remove()
    assert not bucket[key].exists()
    assert key not in bucket
    assert len(bucket) == 0


@parametrize_with_cases("bucket", cases=[bucket_path_bucket])
def test_pathbucket_subdirectory(bucket: PathBucket) -> None:
    subbucket = bucket / "subdir"
    assert subbucket.path.name == "subdir"
    assert subbucket.path.parent == bucket.path
    assert subbucket.path.exists()
