# Buckets
A bucket is a collection of dict-like view of resources that can be accessed by a key
of a given type. This lets you easily store and retrieve objects of varying
types in a single location.

The main implementation we provide is the
[`PathBucket`][byop.store.paths.path_bucket.PathBucket], which is a dict-like
view over a directory to quickly store many files of different types and also
retrieve them.

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

# Store multiple at once
bucket.store(
    {
        "myarray.npy": array,
        "df.csv": dataframe,
        "model.pkl": model,
        "config.json": {"hello": "world"}
    }
)

# Load things
array = bucket["myarray.npy"].load()
maybe_df = bucket["df.csv"].get()  # (3)!
model: LinearRegression = bucket["model.pkl"].get(check=LinearRegression)  # (4)!

# Load multiple at once
items = bucket.fetch("myarray.npy", "df.csv", "model.pkl", "config.json")
array = items["myarray.npy"]
df = items["df.csv"]
model = items["model.pkl"]
config = items["config.json"]

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
