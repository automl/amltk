# Data
AutoML-Toolkit provides some utility for manipulating data containers, specifically
`pd.DataFrame`, `pd.Series`, `np.ndarray`.


## Reducing the size of your data in memory
Often times, the defaults of `numpy` and `pandas` is to use large dtypes that are
suited for most tasks. However sometimes this can be prohibitive, especially in low
memory compute regimes.

To measure the memory consumption of a data container, we can
use [`byte_size()`][amltk.data.byte_size]. While independant methods exist for each of these
structures, we wrap them together in a single function for convenience.

```python exec="true" source="material-block" result="python" title="ref-data-bytesize"
from amltk.data import byte_size

import pandas as pd
import numpy as np

x = np.arange(100)
y = pd.Series(np.linspace(1, 100, 100))
z = pd.DataFrame({"a": np.arange(100), "b": pd.Series(np.linspace(1, 100, 100))})

print("x: ", byte_size(x))
print("y: ", byte_size(y))
print("z: ", byte_size(z))

print("combined: ", byte_size([x, y, z]))
```

Now that we can measure the size of our data, we can use the
[`reduce_dtypes()`][amltk.data.reduce_dtypes] function to reduce the memory of our
data by:

* Find the smallest `int` dtype that can represent integer data
* Reduce the percision of floating point data by one step. i.e. `float64` -> `float32`

```python exec="true" source="material-block" result="python" title="ref-data-reducedtypes"
from amltk.data import reduce_dtypes, byte_size

import pandas as pd
import numpy as np

x = np.arange(100)
y = pd.Series(np.linspace(1, 100, 100))
z = pd.DataFrame({"a": np.arange(100), "b": pd.Series(np.linspace(1, 100, 100))})

print(f"x: {x.dtype}")
print(f"y: {y.dtype}")
print(f"z: {z.dtypes}")

print("combined memory: ", byte_size([x, y, z]))

x, y, z = [reduce_dtypes(d) for d in [x, y, z]]

print(f"x: {x.dtype}")
print(f"y: {y.dtype}")
print(f"z: {z.dtypes}")

print("combined memory: ", byte_size([x, y, z]))
```
