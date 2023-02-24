"""Loaders for [`PathBucket`][byop.store.paths.path_bucket.PathBucket]s.

The [`Loader`][byop.store.loader.Loader]s in this module are used to
load and save objects identified by a unique [`Path`][pathlib.Path].
For saving objects, these loaders rely on checking the type of the
object for [`can_save`][byop.store.loader.Loader.can_save] and
[`save`][byop.store.loader.Loader.save] methods. For loading objects,
these loaders rely on checking the file extension of the path for
[`can_load`][byop.store.loader.Loader.can_load] and
[`load`][byop.store.loader.Loader.load] methods.
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import pickle
from types import ModuleType
from typing import Any, ClassVar, Literal, Protocol, TypeVar

import numpy as np
import pandas as pd

from byop.store.loader import Loader

# NOTE: Since we don't want to depend on yaml, we do a dynamic
# import on it.
yaml: ModuleType | None
try:
    import yaml
except ImportError:
    yaml = None

T = TypeVar("T")


class PathLoader(Loader[Path, T], Protocol[T]):
    """A [`Loader`][byop.store.loader.Loader] for loading and saving
    objects indentified by a [`Path`][pathlib.Path].
    """

    @property
    def name(self) -> str:
        """See [`Loader.name`][byop.store.loader.Loader.name]."""
        ...

    def can_save(self, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        ...

    def can_load(self, path: Path, /, *, check: type[T] | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        ...

    def load(self, path: Path, /) -> T:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        ...

    def save(self, obj: T, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        ...


class NPYLoader(PathLoader[np.ndarray]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`np.ndarray`][numpy.ndarray]s.

    This loader supports the following file extensions:

    * `#!python ".npy"`

    This loader supports the following types:

    * [`np.ndarray`][numpy.ndarray]
    """

    name: ClassVar[Literal["np"]] = "np"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return isinstance(obj, np.ndarray)

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix in {".npy"} and check in (np.ndarray, None)

    @classmethod
    def load(cls, path: Path, /) -> np.ndarray:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        return np.load(path)

    @classmethod
    def save(cls, obj: np.ndarray, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        np.save(path, obj)


class PDLoader(PathLoader[pd.DataFrame]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`pd.DataFrame`][pandas.DataFrame]s.

    This loader supports the following file extensions:

    * `#!python ".csv"`
    * `#!python ".parquet"`

    This loader supports the following types:

    * [`pd.DataFrame`][pandas.DataFrame]

    ???+ note "Multiindex support"
        There is currently no multi-index support as we explicitly
        use `index_col=0` when loading a `.csv` file. This is
        because we assume that the first column is the index to
        prevent Unamed columns from being created.

    ???+ note "Series support"
        There is currently no support for pandas series as once written
        to csv, they are converted to a dataframe with a single column.
        See [this issue](https://github.com/automl/byop/issues/4)
    """

    name: ClassVar[Literal["pd"]] = "pd"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    _load_methods = {
        ".csv": partial(pd.read_csv, index_col=0),
        ".parquet": pd.read_parquet,
    }
    _save_methods = {
        ".csv": partial(pd.DataFrame.to_csv, index=True),
        ".parquet": pd.DataFrame.to_parquet,
    }

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        passes_check = check in (pd.DataFrame, None)
        return path.suffix in cls._load_methods and passes_check

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        if not isinstance(obj, pd.DataFrame):
            return False

        # TODO: https://github.com/automl/byop/issues/4
        if obj.index.nlevels == 1:
            return True

        return False

    @classmethod
    def load(cls, path: Path, /) -> pd.DataFrame:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        load_method = cls._load_methods[path.suffix]
        return load_method(path)

    @classmethod
    def save(cls, obj: pd.Series | pd.DataFrame, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        save_method = cls._save_methods[path.suffix]
        if obj.index.name is None and obj.index.nlevels == 1:
            obj.index.name = "index"
        save_method(obj, path)


class JSONLoader(PathLoader[dict | list]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to JSON.

    This loader supports the following file extensions:

    * `#!python ".json"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar[Literal["json"]] = "json"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix == ".json" and check in (dict, list, None)

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list))

    @classmethod
    def load(cls, path: Path, /) -> dict | list:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        with path.open("r") as f:
            return json.load(f)

    @classmethod
    def save(cls, obj: dict | list, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        with path.open("w") as f:
            json.dump(obj, f)


class YAMLLoader(PathLoader[dict | list]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to YAML.

    This loader supports the following file extensions:

    * `#!python ".yaml"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar[Literal["yaml"]] = "yaml"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix == ".yaml" and check in (dict, list, None)

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list))

    @classmethod
    def load(cls, path: Path, /) -> dict | list:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with path.open("r") as f:
            return yaml.safe_load(f)

    @classmethod
    def save(cls, obj: dict | list, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with path.open("w") as f:
            yaml.dump(obj, f)


class PickleLoader(PathLoader[Any]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving any object to a pickle file.

    This loader supports the following file extensions:

    * `#!python ".pkl"`

    This loader supports the following types:

    * object

    ???+ note "Picklability"
        This loader uses Python's built-in [`pickle`][pickle] module
        to save and load objects. This means that the object must be
        [picklable][pickle] in order to be saved and loaded. If the
        object is not picklable, then an error will be raised when
        attempting to save or load the object.
    """

    name: ClassVar[Literal["pickle"]] = "pickle"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    @classmethod
    def can_load(
        cls, path: Path, /, *, check: type | None = None  # noqa: ARG003
    ) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix in (".pkl", ".pickle")

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:  # noqa: ARG003
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return True  # Anything can be attempted to be pickled

    @classmethod
    def load(cls, path: Path, /) -> Any:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        with path.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def save(cls, obj: Any, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        with path.open("wb") as f:
            pickle.dump(obj, f)


class TxtLoader(PathLoader[str]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`str`][str]s to text files.

    This loader supports the following file extensions:

    * `#!python ".text"`
    * `#!python ".txt"`

    This loader supports the following types:

    * [`str`][str]
    """

    name: ClassVar[Literal["text"]] = "text"
    """See [`Loader.name`][byop.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix in (".text", ".txt") and check in (str, None)

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return isinstance(obj, str)

    @classmethod
    def load(cls, path: Path, /) -> str:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        with path.open("r") as f:
            return f.read()

    @classmethod
    def save(cls, obj: str, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        with path.open("w") as f:
            f.write(obj)


class ByteLoader(PathLoader[bytes]):
    """A [`Loader`][byop.store.loader.Loader] for loading and
    saving [`bytes`][bytes] to binary files.

    This loader supports the following file extensions:

    * `#!python ".bin"`
    * `#!python ".bytes"`

    This loader supports the following types:

    * [`bytes`][bytes]
    """

    name: ClassVar[Literal["bytes"]] = "bytes"

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][byop.store.loader.Loader.can_load]."""
        return path.suffix in (".bin", ".bytes") and check in (bytes, None)

    @classmethod
    def can_save(cls, obj: Any, /) -> bool:
        """See [`Loader.can_save`][byop.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list))

    @classmethod
    def load(cls, path: Path, /) -> bytes:
        """See [`Loader.load`][byop.store.loader.Loader.load]."""
        with path.open("rb") as f:
            return f.read()

    @classmethod
    def save(cls, obj: bytes, path: Path, /) -> None:
        """See [`Loader.save`][byop.store.loader.Loader.save]."""
        with path.open("wb") as f:
            f.write(obj)
