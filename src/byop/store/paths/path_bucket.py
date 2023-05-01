"""A module containing a concreate implementation of a
[`Bucket`][byop.store.bucket.Bucket] that uses the Path API to store objects.
"""
from __future__ import annotations

import shutil
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from byop.store.bucket import Bucket
from byop.store.drop import Drop
from byop.store.paths.path_loaders import (
    ByteLoader,
    JSONLoader,
    NPYLoader,
    PathLoader,
    PDLoader,
    PickleLoader,
    TxtLoader,
    YAMLLoader,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from byop.store.loader import Loader

DEFAULT_LOADERS: tuple[PathLoader, ...] = (
    NPYLoader,
    PDLoader,
    JSONLoader,  # We prefer json over yaml
    YAMLLoader,
    TxtLoader,
    ByteLoader,
    PickleLoader,
)


class PathBucket(Bucket[str, Path]):
    """A bucket that uses the Path API to store objects.

    This bucket is a key-value lookup backed up by some filesystem.
    By assinging to the bucket, you store the object to the filesystem.
    However the values you get back are instead a [`Drop`][byop.store.drop.Drop]
    that can be used to perform operations on the stores object, such as `load`, `get`
    and `remove`.

    ???+ note "Drop methods"
        * [`Drop.load`][byop.store.drop.Drop.load] - Load the object from the bucket.
        * [`Drop.get`][byop.store.drop.Drop.get] - Load the object from the bucket
            with a default if something fails.
        * [`Drop.put`][byop.store.drop.Drop.put] - Store an object in the bucket.
        * [`Drop.remove`][byop.store.drop.Drop.remove] - Remove the object from the
            bucket.
        * [`Drop.exists`][byop.store.drop.Drop.exists] - Check if the object exists
            in the bucket.

    ```python
    from byop.store.paths import PathBucket
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    bucket = PathBucket("path/to/bucket")

    array = np.array([1, 2, 3])
    dataframe = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    model = LinearRegression()

    # Store things
    bucket["myarray.npy"] = array # (1)!
    bucket["df.csv"] = dataframe  # (2)!
    bucket["model.pkl"].put(model)

    bucket["config.json"] = {"hello": "world"}
    assert bucket["config.json"].exists()
    bucket["config.json"].remove()

    # Load things
    array = bucket["myarray.npy"].load()
    maybe_df = bucket["df.csv"].get()  # (3)!
    model: LinearRegression = bucket["model.pkl"].get(check=LinearRegression)  # (4)!

    # Create subdirectories
    model_bucket = bucket / "my_model" # (5)!
    model_bucket["model.pkl"] = model
    model_bucket["predictions.npy"] = model.predict(X)

    # Acts like a mapping
    assert "myarray.npy" in bucket
    assert len(bucket) == 3
    for key, item in bucket.items():
        print(key, item.load())
    del bucket["model.pkl"]
    ```

    1. The `=` is a shortcut for `bucket["myarray.npy"].put(array)`
    2. The extension is used to determine which
        [`PathLoader`][byop.store.paths.path_loaders.PathLoader] to use
        and how to save it.
    3. The `get` method acts like the [`dict.load`][dict] method.
    4. The `get` method can be used to check the type of the loaded object.
        If the type does not match, a `TypeError` is raised.
    5. Uses the familiar [`Path`][pathlib.Path] API to create subdirectories.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        loaders: Sequence[Loader[Path, Any]] | None = None,
        create: bool = True,
        clean: bool = False,
        exists_ok: bool = True,
    ) -> None:
        """Create a new PathBucket.

        Args:
            path: The path to the bucket.
            loaders: A sequence of loaders to use when loading objects.
                These will be prepended to the default loaders and attempted
                to be used first.
            create: If True, the base path will be created if it does not
                exist.
            clean: If True, the base path will be deleted if it exists.
            exists_ok: If False, an error will be raised if the base path
                already exists.
        """
        _loaders = DEFAULT_LOADERS
        if loaders is not None:
            _loaders = tuple(chain(loaders, DEFAULT_LOADERS))

        if isinstance(path, str):
            path = Path(path)

        if clean and path.exists():
            shutil.rmtree(path, ignore_errors=True)

        if not exists_ok and path.exists():
            raise FileExistsError(f"File/Directory already exists at {path}")

        if create:
            path.mkdir(parents=True, exist_ok=True)

        self._create = create
        self.path = path
        self.loaders = _loaders

    def __getitem__(self, key: str) -> Drop[Path]:
        return self._drop(self.path / key, loaders=self.loaders)

    def __setitem__(self, key: str, value: Any) -> None:
        self._drop(self.path / key, loaders=self.loaders).put(value)

    def __delitem__(self, key: str) -> None:
        self._drop(self.path / key, loaders=self.loaders).remove()

    def __iter__(self) -> Iterator[str]:
        return (path.name for path in self.path.iterdir())

    def sub(self, key: str, *, create: bool | None = None) -> Self:
        """Create a subdirectory of the bucket.

        Args:
            key: The name of the subdirectory.
            create: Whether the subdirectory will be created if it does not
                exist. If None, the default, the value of `create` passed to
                the constructor will be used.

        Returns:
            A new bucket with the same loaders as the current bucket.
        """
        return self.__class__(
            self.path / key,
            loaders=self.loaders,
            create=self._create if create is None else create,
            clean=False,
        )

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return (self.path / key).exists()

    @classmethod
    def _drop(cls, path: Path, loaders: tuple[PathLoader, ...]) -> Drop[Path]:
        return Drop(
            path,
            loaders=loaders,
            _remove=cls._remove,
            _exists=cls._exists,
        )

    @classmethod
    def _remove(cls, path: Path) -> None:
        if path.is_dir():
            for child in path.iterdir():
                cls._remove(child)
            path.rmdir()
        else:
            path.unlink()

    @classmethod
    def _exists(cls, path: Path) -> bool:
        return path.exists()
