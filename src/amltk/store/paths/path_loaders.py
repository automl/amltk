"""Loaders for [`PathBucket`][amltk.store.paths.path_bucket.PathBucket]s.

The [`Loader`][amltk.store.paths.path_loaders.PathLoader]s in this module are used to
load and save objects identified by a unique [`Path`][pathlib.Path].
For saving objects, these loaders rely on checking the type of the
object for [`can_save`][amltk.store.paths.path_loaders.PathLoader.can_save] and
[`save`][amltk.store.paths.path_loaders.PathLoader.save] methods. For loading objects,
these loaders rely on checking the file extension of the path for
[`can_load`][amltk.store.paths.path_loaders.PathLoader.can_load] and
[`load`][amltk.store.paths.path_loaders.PathLoader.load] methods.
"""
from __future__ import annotations

import json
import logging
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from typing_extensions import override

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


class PathLoader(Loader[Path, T]):
    """A [`Loader`][amltk.store.loader.Loader] for loading and saving
    objects indentified by a [`Path`][pathlib.Path].
    """

    name: ClassVar[str]
    """The name of the loader."""

    @override
    @classmethod
    @abstractmethod
    def can_load(cls, key: Path, /, *, check: type[T] | None = None) -> bool:
        """Return True if this loader supports the resource at key.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            key: The key used to identify the resource
            check: If the loader can support loading a specific type
                of object.
        """
        ...

    @override
    @classmethod
    @abstractmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """Return True if this loader can save this object.

        This is used to determine which loader to use when loading a
        resource from a key.

        Args:
            obj: The object to save.
            key: The key used to identify the resource
        """
        ...

    @override
    @classmethod
    @abstractmethod
    def save(cls, obj: Any, key: Path, /) -> None:
        """Save an object to under the given key.

        Args:
            obj: The object to save.
            key: The key to save the object under.
        """
        ...

    @override
    @classmethod
    @abstractmethod
    def load(cls, key: Path, /) -> T:
        """Load an object from the given key.

        Args:
            key: The key to load the object from.

        Returns:
            The loaded object.
        """
        ...


class NPYLoader(PathLoader[np.ndarray]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`np.ndarray`][numpy.ndarray]s.

    This loader supports the following file extensions:

    * `#!python ".npy"`

    This loader supports the following types:

    * [`np.ndarray`][numpy.ndarray]
    """

    name: ClassVar = "np"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return isinstance(obj, np.ndarray) and key.suffix in {".npy"}

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix in {".npy"} and check in (np.ndarray, None)

    @override
    @classmethod
    def load(cls, key: Path, /) -> np.ndarray:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        item = np.load(key, allow_pickle=False)
        if not isinstance(item, np.ndarray):
            msg = f"Expected `np.ndarray` from {key=} but got `{type(item).__name__}`."
            raise TypeError(msg)

        return item

    @override
    @classmethod
    def save(cls, obj: np.ndarray, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        np.save(key, obj, allow_pickle=False)


class PDLoader(PathLoader[pd.DataFrame | pd.Series]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`pd.DataFrame`][pandas.DataFrame]s.

    This loader supports the following file extensions:

    * `#!python ".csv"`
    * `#!python ".parquet"`
    * `#!python ".pdpickle"`

    This loader supports the following types:

    * [`pd.DataFrame`][pandas.DataFrame]
    * [`pd.Series`][pandas.Series] - Only to `#!python ".pdpickle"` files

    ???+ note "Multiindex support"

        There is currently no multi-index support as we explicitly
        use `index_col=0` when loading a `".csv"` file. This is
        because we assume that the first column is the index to
        prevent Unamed columns from being created.

    ???+ note "Series support"

        There is currently limited support for pandas series as once written
        to csv/parquet, they are converted to a dataframe with a single column.
        See [this issue](https://github.com/automl/amltk/issues/4)

        Please consider using `".pdpickle"` instead.
    """

    name: ClassVar = "pd"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        if key.suffix in (".pdpickle", None):
            return check in (pd.Series, pd.DataFrame, None)

        if key.suffix in (".csv", ".parquet"):
            return check in (pd.DataFrame, None)

        return False

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        if key.suffix == ".pdpickle":
            return isinstance(obj, pd.Series | pd.DataFrame)

        if key.suffix == ".parquet":
            return isinstance(obj, pd.DataFrame)

        if key.suffix == ".csv":
            # TODO: https://github.com/automl/amltk/issues/4
            return isinstance(obj, pd.DataFrame) and obj.index.nlevels == 1

        return False

    @override
    @classmethod
    def load(cls, key: Path, /) -> pd.DataFrame | pd.Series:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        if key.suffix == ".csv":
            return pd.read_csv(key, index_col=0)

        if key.suffix == ".parquet":
            return pd.read_parquet(key)

        if key.suffix == ".pdpickle":
            obj = pd.read_pickle(key)  # noqa: S301
            if not isinstance(obj, pd.Series | pd.DataFrame):
                msg = (
                    f"Expected `pd.Series | pd.DataFrame` from {key=}"
                    f" but got `{type(obj).__name__}`."
                )
                raise TypeError(msg)

            return obj

        raise ValueError(f"Unknown file extension {key.suffix}")

    @override
    @classmethod
    def save(cls, obj: pd.Series | pd.DataFrame, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        # Most pandas methods only seem to support dataframes
        logger.debug(f"Saving {key=}")

        if key.suffix == ".pdpickle":
            obj.to_pickle(key)
            return

        if key.suffix == ".csv":
            if obj.index.name is None and obj.index.nlevels == 1:
                obj.index.name = "index"

            obj.to_csv(key, index=True)
            return

        if key.suffix == ".parquet":
            obj.to_parquet(key)
            return

        raise ValueError(f"Unknown extension {key.suffix=}")


class JSONLoader(PathLoader[dict | list]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to JSON.

    This loader supports the following file extensions:

    * `#!python ".json"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar = "json"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix == ".json" and check in (dict, list, None)

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return isinstance(obj, dict | list) and key.suffix == ".json"

    @override
    @classmethod
    def load(cls, key: Path, /) -> dict | list:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        with key.open("r") as f:
            item = json.load(f)

        if not isinstance(item, dict | list):
            msg = f"Expected `dict | list` from {key=} but got `{type(item).__name__}`"
            raise TypeError(msg)

        return item

    @override
    @classmethod
    def save(cls, obj: dict | list, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        with key.open("w") as f:
            json.dump(obj, f)


class YAMLLoader(PathLoader[dict | list]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`dict`][dict]s and [`list`][list]s to YAML.

    This loader supports the following file extensions:

    * `#!python ".yaml"`
    * `#!python ".yml"`

    This loader supports the following types:

    * [`dict`][dict]
    * [`list`][list]
    """

    name: ClassVar = "yaml"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix in (".yaml", ".yml") and check in (dict, list, None)

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return isinstance(obj, dict | list) and key.suffix in (".yaml", ".yml")

    @override
    @classmethod
    def load(cls, key: Path, /) -> dict | list:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with key.open("r") as f:
            item = yaml.safe_load(f)

        if not isinstance(item, dict | list):
            msg = f"Expected `dict | list` from {key=} but got `{type(item).__name__}`"
            raise TypeError(msg)

        return item

    @override
    @classmethod
    def save(cls, obj: dict | list, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        if yaml is None:
            raise ModuleNotFoundError("PyYAML is not installed")

        with key.open("w") as f:
            yaml.dump(obj, f)


class PickleLoader(PathLoader[Any]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
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

    name: ClassVar = "pickle"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_load(
        cls,
        key: Path,
        /,
        *,
        check: type | None = None,
    ) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix in (".pkl", ".pickle")

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return True  # Anything can be attempted to be pickled

    @override
    @classmethod
    def load(cls, key: Path, /) -> Any:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        with key.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    @override
    @classmethod
    def save(cls, obj: Any, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        with key.open("wb") as f:
            pickle.dump(obj, f)


class TxtLoader(PathLoader[str]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`str`][str]s to text files.

    This loader supports the following file extensions:

    * `#!python ".text"`
    * `#!python ".txt"`

    This loader supports the following types:

    * [`str`][str]
    """

    name: ClassVar = "text"
    """::: amltk.store.paths.path_loaders.PathLoader.name"""

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix in (".text", ".txt") and check in (str, None)

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return isinstance(obj, str) and key.suffix in (".text", ".txt")

    @override
    @classmethod
    def load(cls, key: Path, /) -> str:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        with key.open("r") as f:
            return f.read()

    @override
    @classmethod
    def save(cls, obj: str, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        with key.open("w") as f:
            f.write(obj)


class ByteLoader(PathLoader[bytes]):
    """A [`Loader`][amltk.store.paths.path_loaders.PathLoader] for loading and
    saving [`bytes`][bytes] to binary files.

    This loader supports the following file extensions:

    * `#!python ".bin"`
    * `#!python ".bytes"`

    This loader supports the following types:

    * [`bytes`][bytes]
    """

    name: ClassVar = "bytes"

    @override
    @classmethod
    def can_load(cls, key: Path, /, *, check: type | None = None) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_load"""  # noqa: D415
        return key.suffix in (".bin", ".bytes") and check in (bytes, None)

    @override
    @classmethod
    def can_save(cls, obj: Any, key: Path, /) -> bool:
        """::: amltk.store.paths.path_loaders.PathLoader.can_save"""  # noqa: D415
        return isinstance(obj, dict | list) and key.suffix in (".bin", ".bytes")

    @override
    @classmethod
    def load(cls, key: Path, /) -> bytes:
        """::: amltk.store.paths.path_loaders.PathLoader.load"""  # noqa: D415
        logger.debug(f"Loading {key=}")
        with key.open("rb") as f:
            return f.read()

    @override
    @classmethod
    def save(cls, obj: bytes, key: Path, /) -> None:
        """::: amltk.store.paths.path_loaders.PathLoader.save"""  # noqa: D415
        logger.debug(f"Saving {key=}")
        with key.open("wb") as f:
            f.write(obj)
