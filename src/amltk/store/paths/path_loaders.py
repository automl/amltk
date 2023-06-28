"""Loaders for [`PathBucket`][amltk.store.paths.path_bucket.PathBucket]s.

The [`Loader`][amltk.store.loader.Loader]s in this module are used to
load and save objects identified by a unique [`Path`][pathlib.Path].
For saving objects, these loaders rely on checking the type of the
object for [`can_save`][amltk.store.loader.Loader.can_save] and
[`save`][amltk.store.loader.Loader.save] methods. For loading objects,
these loaders rely on checking the file extension of the path for
[`can_load`][amltk.store.loader.Loader.can_load] and
[`load`][amltk.store.loader.Loader.load] methods.
"""
from __future__ import annotations

import json
import logging
import pickle
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from amltk.store.loader import Loader

if TYPE_CHECKING:
    from types import ModuleType

# NOTE: Since we don't want to depend on yaml, we do a dynamic
# import on it.
yaml: ModuleType | None
try:
    import yaml
except ImportError:
    yaml = None

T = TypeVar("T")

logger = logging.getLogger(__name__)


class PathLoader(Loader[Path, T], Protocol[T]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and saving
    objects indentified by a [`Path`][pathlib.Path].
    """

    @property
    def name(self) -> str:
        """See [`Loader.name`][amltk.store.loader.Loader.name]."""
        ...

    def can_save(self, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        ...

    def can_load(self, path: Path, /, *, check: type[T] | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        ...

    def load(self, path: Path, /) -> T:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        ...

    def save(self, obj: T, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        ...


class NPYLoader(PathLoader[np.ndarray]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
    saving [`np.ndarray`][numpy.ndarray]s.

    This loader supports the following file extensions:

    * `#!python ".npy"`

    This loader supports the following types:

    * [`np.ndarray`][numpy.ndarray]
    """

    name: ClassVar[Literal["np"]] = "np"
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return isinstance(obj, np.ndarray) and path.suffix in {".npy"}

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix in {".npy"} and check in (np.ndarray, None)

    @classmethod
    def load(cls, path: Path, /) -> np.ndarray:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        item = np.load(path)
        if not isinstance(item, np.ndarray):
            msg = f"Expected `np.ndarray` from {path=} but got `{type(item).__name__}`."
            raise TypeError(msg)

        return item

    @classmethod
    def save(cls, obj: np.ndarray, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        np.save(path, obj)


class PDLoader(PathLoader[pd.DataFrame]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
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
        See [this issue](https://github.com/automl/amltk/issues/4)
    """

    name: ClassVar[Literal["pd"]] = "pd"
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    _load_methods: Mapping[str, Callable] = {
        ".csv": partial(pd.read_csv, index_col=0),
        ".parquet": pd.read_parquet,
    }
    _save_methods: Mapping[str, Callable] = {
        ".csv": partial(pd.DataFrame.to_csv, index=True),
        ".parquet": pd.DataFrame.to_parquet,
    }

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        passes_check = check in (pd.DataFrame, None)
        return path.suffix in cls._load_methods and passes_check

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        if path.suffix not in cls._save_methods:
            return False
        if not isinstance(obj, pd.DataFrame):
            return False

        # TODO: https://github.com/automl/amltk/issues/4
        if obj.index.nlevels == 1:
            return True

        return False

    @classmethod
    def load(cls, path: Path, /) -> pd.DataFrame:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        load_method = cls._load_methods[path.suffix]
        return load_method(path)

    @classmethod
    def save(cls, obj: pd.Series | pd.DataFrame, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        save_method = cls._save_methods[path.suffix]
        if obj.index.name is None and obj.index.nlevels == 1:
            obj.index.name = "index"
        save_method(obj, path)


class JSONLoader(PathLoader[Union[dict, list]]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to JSON.

    This loader supports the following file extensions:

    * `#!python ".json"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar[Literal["json"]] = "json"
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix == ".json" and check in (dict, list, None)

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list)) and path.suffix == ".json"

    @classmethod
    def load(cls, path: Path, /) -> dict | list:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        with path.open("r") as f:
            item = json.load(f)

        if not isinstance(item, (dict, list)):
            msg = f"Expected `dict | list` from {path=} but got `{type(item).__name__}`"
            raise TypeError(msg)

        return item

    @classmethod
    def save(cls, obj: dict | list, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        with path.open("w") as f:
            json.dump(obj, f)


class YAMLLoader(PathLoader[Union[dict, list]]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to YAML.

    This loader supports the following file extensions:

    * `#!python ".yaml"`
    * `#!python ".yml"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar[Literal["yaml"]] = "yaml"
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix in (".yaml", ".yml") and check in (dict, list, None)

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list)) and path.suffix in (".yaml", ".yml")

    @classmethod
    def load(cls, path: Path, /) -> dict | list:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with path.open("r") as f:
            item = yaml.safe_load(f)

        if not isinstance(item, (dict, list)):
            msg = f"Expected `dict | list` from {path=} but got `{type(item).__name__}`"
            raise TypeError(msg)

        return item

    @classmethod
    def save(cls, obj: dict | list, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with path.open("w") as f:
            yaml.dump(obj, f)


class PickleLoader(PathLoader[Any]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
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
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    @classmethod
    def can_load(
        cls,
        path: Path,
        /,
        *,
        check: type | None = None,  # noqa: ARG003
    ) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix in (".pkl", ".pickle")

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:  # noqa: ARG003
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return True  # Anything can be attempted to be pickled

    @classmethod
    def load(cls, path: Path, /) -> Any:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    @classmethod
    def save(cls, obj: Any, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        with path.open("wb") as f:
            pickle.dump(obj, f)


class TxtLoader(PathLoader[str]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
    saving [`str`][str]s to text files.

    This loader supports the following file extensions:

    * `#!python ".text"`
    * `#!python ".txt"`

    This loader supports the following types:

    * [`str`][str]
    """

    name: ClassVar[Literal["text"]] = "text"
    """See [`Loader.name`][amltk.store.loader.Loader.name]."""

    @classmethod
    def can_load(cls, path: Path, /, *, check: type | None = None) -> bool:
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix in (".text", ".txt") and check in (str, None)

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return isinstance(obj, str) and path.suffix in (".text", ".txt")

    @classmethod
    def load(cls, path: Path, /) -> str:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        with path.open("r") as f:
            return f.read()

    @classmethod
    def save(cls, obj: str, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        with path.open("w") as f:
            f.write(obj)


class ByteLoader(PathLoader[bytes]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and
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
        """See [`Loader.can_load`][amltk.store.loader.Loader.can_load]."""
        return path.suffix in (".bin", ".bytes") and check in (bytes, None)

    @classmethod
    def can_save(cls, obj: Any, path: Path, /) -> bool:
        """See [`Loader.can_save`][amltk.store.loader.Loader.can_save]."""
        return isinstance(obj, (dict, list)) and path.suffix in (".bin", ".bytes")

    @classmethod
    def load(cls, path: Path, /) -> bytes:
        """See [`Loader.load`][amltk.store.loader.Loader.load]."""
        logger.debug(f"Loading {path=}")
        with path.open("rb") as f:
            return f.read()

    @classmethod
    def save(cls, obj: bytes, path: Path, /) -> None:
        """See [`Loader.save`][amltk.store.loader.Loader.save]."""
        logger.debug(f"Saving {path=}")
        with path.open("wb") as f:
            f.write(obj)
