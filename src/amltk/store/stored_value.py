"""A value that is stored on disk and loaded lazily.

This is useful for transmitting large objects between processes.

```python exec="true" source="material-block" result="python" title="StoredValue"
from amltk.store import StoredValue
import pandas as pd
from pathlib import Path

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
path = Path("df.csv")
df.to_csv(path)

stored_value = StoredValue(path, read=pd.read_csv)

# Somewhere in a processes
df = stored_value.value()
print(df)

path.unlink()
```

You can quickly obtain these from buckets if you require
```python exec="true" source="material-block" result="python" title="StoredValue from bucket"
from amltk import PathBucket
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
bucket = PathBucket("bucket_path")
bucket.update({"df.csv": df})

stored_value = bucket["df.csv"].as_stored_value()

# Somewhere in a processes
df = stored_value.value()
print(df)

bucket.rmdir()
```
"""  # noqa: E501
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class StoredValue(Generic[K, V]):
    """A value that is stored on disk and can be loaded when needed."""

    key: K
    read: Callable[[K], V]

    _value: V | None = None

    def value(self) -> V:
        """Get the value."""
        if self._value is None:
            self._value = self.read(self.key)

        return self._value
