"""A value that is stored on disk and loaded lazily.

This is useful for transmitting large objects between processes.

```python exec="true" source="material-block" result="python" title="Stored"
from amltk.store import Stored
import pandas as pd
from pathlib import Path

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
path = Path("df.csv")

df.to_csv(path)
stored_df = Stored(path, read=pd.read_csv)

# Somewhere in a processes
df = stored_df.load()
print(df)
path.unlink()  # markdown-exec: hide
```

You can quickly obtain these from buckets if you require using
[`put()`][amltk.store.drop.Drop.put].

```python exec="true" source="material-block" result="python" title="Stored from bucket"
from amltk import PathBucket
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
bucket = PathBucket("bucket_path")

stored_df = bucket["df.csv"].put(df)

# Somewhere in a processes
df = stored_df.load()
print(df)
bucket.rmdir()  # markdown-exec: hide
```
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Stored(Generic[V]):
    """A value that is stored on disk and can be loaded when needed."""

    def __init__(self, key: K, read: Callable[[K], V]):
        """Initialize the stored value.

        Args:
            key: The key to load the value from.
            read: A function that takes a key and returns the value.
        """
        super().__init__()
        self.key = key
        self.read = read

    def load(self) -> V:
        """Get the value."""
        return self.read(self.key)
