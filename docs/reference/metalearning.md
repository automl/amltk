# Metalearning
An important part of AutoML systems is to perform well on new unseen data.
There are a variety of methods to do so but we provide some building blocks
to help implement these methods.

## MetaFeatures
Calculating meta-features of a dataset is quite straight foward.

```python exec="true" source="material-block" result="python" title="Metafeatures" hl_lines="10"
import openml
from amltk.metalearning import compute_metafeatures

dataset = openml.datasets.get_dataset(31)  # credit-g
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)

mfs = compute_metafeatures(X, y)

print(mfs)
```

By default [`compute_metafeatures()`][amltk.metalearning.compute_metafeatures] will
calculate all the [`MetaFeature`][amltk.metalearning.MetaFeature] implemented,
iterating through their subclasses to do so. You can pass an explicit list
as well to `compute_metafeatures(X, y, features=[...])`.

To implement your own is also quite straight forward:

```python exec="true" source="material-block" result="python" title="Create Metafeature" hl_lines="10 11 12 13 14 15 16 17 18 19"
from amltk.metalearning import MetaFeature, compute_metafeatures
import openml

dataset = openml.datasets.get_dataset(31)  # credit-g
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)

class TotalValues(MetaFeature):

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> int:
        return int(x.shape[0] * x.shape[1])

mfs = compute_metafeatures(X, y, features=[TotalValues])
print(mfs)
```

As many metafeatures rely on pre-computed dataset statistics, and they do not
need to be calculated more than once, you can specify the dependancies of
a meta feature. When a metafeature would return something other than a single
value, i.e. a `dict` or a `pd.DataFrame`, we instead call those a
[`DatasetStatistic`][amltk.metalearning.DatasetStatistic]. These will
**not** be included in the result of [`compute_metafeatures()`][amltk.metalearning.compute_metafeatures].
These `DatasetStatistic`s will only be calculated once on a call to `compute_metafeatures()` so
they can be re-used across all `MetaFeature`s that require that dependancy.

```python exec="true" source="material-block" result="python" title="Metafeature Dependancy" hl_lines="10 11 12 13 14 15 16 17 18 19 20 23 26 35"
from amltk.metalearning import MetaFeature, DatasetStatistic, compute_metafeatures
import openml

dataset = openml.datasets.get_dataset(31)  # credit-g
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute,
)

class NAValues(DatasetStatistic):
    """A mask of all NA values in a dataset"""

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> pd.DataFrame:
        return x.isna()


class PercentageNA(MetaFeature):
    """The percentage of values missing"""

    dependencies = (NAValues,)

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> int:
        na_values = dependancy_values[NAValues]
        n_na = na_values.sum().sum()
        n_values = int(x.shape[0] * x.shape[1])
        return float(n_na / n_values)

mfs = compute_metafeatures(X, y, features=[PercentageNA])
print(mfs)
```

To view the description of a particular `MetaFeature`, you can call
[`.description()`][amltk.metalearning.DatasetStatistic.description]
on it. Otherwise you can access all of them in the following way:

```python exec="true" source="tabbed-left" result="python" title="Metafeature Descriptions" hl_lines="4"
from pprint import pprint
from amltk.metalearning import metafeature_descriptions

descriptions = metafeature_descriptions()
for name, description in descriptions.items():
    print("---")
    print(name)
    print("---")
    print(" * " + description)
```

## Dataset Distances
One common way to define how similar two datasets are is to compute some "similarity"
between them. This notion of "similarity" requires computing some features of a dataset
(**metafeatures**) first, such that we can numerically compute some distance function.

Let's see how we can quickly compute the distance between some datasets with
[`dataset_distance()`][amltk.metalearning.dataset_distance]!

```python exec="true" source="material-block" result="python" title="Dataset Distances P.1" session='dd'
import pandas as pd
import openml

from amltk.metalearning import compute_metafeatures

def get_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    return X, y

d31 = get_dataset(31)
d3 = get_dataset(3)
d4 = get_dataset(4)

metafeatures_dict = {
    "dataset_31": compute_metafeatures(*d31),
    "dataset_3": compute_metafeatures(*d3),
    "dataset_4": compute_metafeatures(*d4),
}

metafeatures = pd.DataFrame(metafeatures_dict)
print(metafeatures)
```

Now we want to know which one of `#!python "dataset_3"` or `#!python "dataset_4"` is
more _similar_ to `#!python "dataset_31"`.

```python exec="true" source="material-block" result="python" title="Dataset Distances P.2" session='dd'
from amltk.metalearning import dataset_distance

target = metafeatures_dict.pop("dataset_31")
others = metafeatures_dict

distances = dataset_distance(target, others, distance_metric="l2")
print(distances)
```

Seems like `#!python "dataset_3"` is some notion of closer to `#!python "dataset_31"`
than `#!python "dataset_4"`. However the scale of the metafeatures are not exactly all close.
For example, many lie between `#!python (0, 1)` but some like `instance_count` can completely
dominate the show.

Lets repeat the computation but specify that we should apply a `#!python "minmax"` scaling
across the rows.

```python exec="true" source="material-block" result="python" title="Dataset Distances P.3" session='dd' hl_lines="5"
distances = dataset_distance(
    target,
    others,
    distance_metric="l2",
    scaler="minmax"
)
print(distances)
```

While `#!python "dataset_3"` is still considered more similar, the difference between the two is a lot less
dramatic. In general, applying some scaling to values of different scales is required for metalearning.

You can also use an [sklearn.preprocessing.MinMaxScaler][] or anything other scaler from scikit-learn
for that matter.

```python exec="true" source="material-block" result="python" title="Dataset Distances P.3" session='dd' hl_lines="7"
from sklearn.preprocessing import MinMaxScaler

distances = dataset_distance(
    target,
    others,
    distance_metric="l2",
    scaler=MinMaxScaler()
)
print(distances)
```

## Portfolio Selection
Another common trick in meta-learning is to define a portfolio of configurations that maximize some
notion of converage across those datasets. The intution here is that this also means that any
new dataset is also covered!

Suppose we hade the given performances of some configurations across some datasets.
```python exec="true" source="material-block" result="python" title="Initial Portfolio"
import pandas as pd

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])
print(portfolio)
```

If we could only choose `#!python k=3` of these configurations on some new given dataset, which ones would
you choose and in what priority?
Here is where we can apply [`portfolio_selection()`][amltk.metalearning.portfolio_selection]!

The idea is that we pick a subset of these algorithms that maximise some value of utility for
the portfolio. We do this by adding a single configuration from the entire set, 1-by-1 until
we reach `k`, beggining with the empty portfolio.

Let's see this in action!

```python exec="true" source="material-block" result="python" title="Portfolio Selection" hl_lines="12 13 14 15 16"
import pandas as pd
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax"
)

print(selected_portfolio)
print()
print(trajectory)
```

The trajectory tells us which configuration was added at each time stamp along with the utility
of the portfolio with that configuration added. However we havn't specified how _exactly_ we defined the
utility of a given portfolio. We could define our own function to do so:

```python exec="true" source="material-block" result="python" title="Portfolio Selection Custom" hl_lines="12 13 14 20"
import pandas as pd
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

def my_function(p: pd.DataFrame) -> float:
    """Take the maximum score for each dataset and then take the mean across them."""
    return p.max(axis=1).mean()

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax",
    portfolio_value=my_function,
)

print(selected_portfolio)
print()
print(trajectory)
```

This notion of reducing across all configurations for a dataset and then aggregating these is common
enough that we can also directly just define these operations and we will perform the rest.

```python exec="true" source="material-block" result="python" title="Portfolio Selection With Reduction" hl_lines="17 18"
import pandas as pd
import numpy as np
from amltk.metalearning import portfolio_selection

performances = {
    "c1": [90, 60, 20, 10],
    "c2": [20, 10, 90, 20],
    "c3": [10, 20, 40, 90],
    "c4": [90, 10, 10, 10],
}
portfolio = pd.DataFrame(performances, index=["dataset_1", "dataset_2", "dataset_3", "dataset_4"])

selected_portfolio, trajectory = portfolio_selection(
    portfolio,
    k=3,
    scaler="minmax",
    row_reducer=np.max,  # This is actually the default
    aggregator=np.mean,  # This is actually the default
)

print(selected_portfolio)
print()
print(trajectory)
```
