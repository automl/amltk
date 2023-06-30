# Scikit-learn
Scikit-learn is a library for classical machine learning,
implementing many of the time-tested, non-deep, methods for
machine learning. It includes many models, hyperparameters
for these models and is its own toolkit for evaluating
these models.

We extend these capabilities with what we found helpful
during development of AutoML tools such as
[AutoSklearn](https://automl.github.io/auto-sklearn/master/).

!!! note "Threads and Multiprocessing"

    If running multiple trainings across multiple processes, please
    also check out [`ThreadPoolCTL`](../reference/threadpoolctl.md)

## Pipeline Builder
The `amltk.sklearn` module provides a `build_pipeline` function
that can be passed to [`Pipeline.build()`][amltk.pipeline.Pipeline.build]
to create a pure [sklearn.pipeline.Pipeline][] from your definition.

### A simple Pipeline

```python exec="true" source="material-block" result="python" title="A simple Pipeline" hl_lines="12"
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from amltk.pipeline import step, Pipeline
from amltk.sklearn import sklearn_pipeline

pipeline = Pipeline.create(
    step("imputer", SimpleImputer, config={"strategy": "median"}),
    step("rf", RandomForestClassifier, config={"n_estimators": 10}),
)

sklearn_pipeline = pipeline.build(builder=sklearn_pipeline)
print(sklearn_pipeline)
```

!!! note "Implicit building"

    By default, AutoML-Toolkit will try to infer how to build your
    pipeline when you call [`Pipeline.build()`][amltk.pipeline.Pipeline.build].
    If all the components contained in the `Pipeline` are from
    `sklearn`, then it will use the `sklearn_pipeline` automatically.

    You will rarely have to explicitly pass the `builder` argument.

### Data Preprocessing
Below is a fairly complex pipeline which handles data-preprocessing,
feeding `#!python "categoricals"` through a
[SimpleImputer][sklearn.impute.SimpleImputer] and a
[OneHotEncoder][sklearn.preprocessing.OneHotEncoder] and
`#!python "numerics"` through a
[SimpleImputer][sklearn.impute.SimpleImputer],
[VarianceThreshold][sklearn.feature_selection.VarianceThreshold]
and possibly a [StandardScaler][sklearn.preprocessing.StandardScaler].

This is done using the [`split()`][amltk.pipeline.split] operator
from AutoML-toolkit, which allows you to split your data into
multiple branches and then combine them back together.

You will notice for `#!python "feature_preprocessing"` split, we
pass the `item=` as a [ColumnTransformer][sklearn.compose.ColumnTransformer]
and for the `config=` parameter, two [make_column_selector][sklearn.compose.make_column_selector]
functions whose names match those of the two split paths, `#!python "categoricals"`
and `#!python "numerics"`.

!!! quote "No Custom `amltk` Components"

    To keep things as compatible as possible with `sklearn`, we
    do not provide any custom components. This lets use export
    things easily and allows you to include your own sklearn
    components in your pipeline without us getting in the way.


```python exec="true" source="material-block" result="python" title="A complex Pipeline" hl_lines="53 54 55 56 57"
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

from amltk.sklearn import sklearn_pipeline
from amltk.pipeline import step, split, choice, group, Pipeline

pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        group(
            "categoricals",
            step(
                "categorical_imputer",
                SimpleImputer,
                space={
                    "strategy": ["most_frequent", "constant"],
                    "fill_value": ["missing"],
                },
            )
            | step(
                "ohe",
                OneHotEncoder,
                space={
                    "min_frequency": (0.01, 0.1),
                    "handle_unknown": ["ignore", "infrequent_if_exist"],
                },
                config={"drop": "first"},
            )
        ),
        group(
            "numericals",
            step("numerical_imputer", SimpleImputer, space={"strategy": ["mean", "median"]})
            | step(
                "variance_threshold",
                VarianceThreshold,
                space={"threshold": (0.0, 0.2)},
            )
            | choice(
                "scaler",
                step("standard", StandardScaler),
                step("minmax", MinMaxScaler),
                step("passthrough", FunctionTransformer),
            )
        ),
        item=ColumnTransformer,
        config={
            "categoricals": make_column_selector(dtype_include=object),
            "numericals": make_column_selector(dtype_include=np.number),
        },
    ),
    choice(
        "algorithm",
        step("svm", SVC, space={"C": (0.1, 10.0)}, config={"probability": True}),
        step(
            "rf",
            RandomForestClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),
    )
)

space = pipeline.space()
config = pipeline.sample(space)
configured_pipeline = pipeline.configure(config)

# `builder=` is optional, we can detect it's an sklearn pipeline.
sklearn_pipeline = configured_pipeline.build(builder=sklearn_pipeline)
print(sklearn_pipeline)
```

## Data Splitting
We also provide two convenience functions often required in AutoML
systems, namely [`train_val_test_split()`][amltk.sklearn.train_val_test_split]
for creating three splits of your data and
[`split_data()`][amltk.sklearn.split_data] for creating an arbitrary number
of splits.

### Train, Val, Test Split
This functions much similar to the sklearn
[`train_test_split()`][sklearn.model_selection.train_test_split] but produces one more
split, the validation split.

Instead of passing in a `test_size=` parameter, you pass in a
`splits=` parameter, which declares the percentages of splits you
would like, e.g. `(0.5, 0.3, 0.2)` would indicate a train size of `50%`,
a val size of `30%` and a test size of `20%`.

```python exec="true" source="material-block" result="python" title="Train, Val, Test Split"
from amltk.sklearn.data import train_val_test_split

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(
    x, y, splits=(0.5, 0.3, 0.2), seed=42
)

print(train_x, train_y)
print(val_x, val_y)
print(test_x, test_y)
```

You may also use the `shuffle=` and `stratify=` parameters to
shuffle and stratify your data respectively. The `stratify=` argument
will respect the stratification across all 3 splits, ensuring they each
have a proportionate amount of each value in `stratify=`.

```python exec="true" source="material-block" result="python" title="Train, Val, Test Split with Shuffle and Stratify" hl_lines="10 11"

from amltk.sklearn.data import train_val_test_split

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(
    x, y,
    splits=(0.5, 0.3, 0.2),
    stratify=y,
    shuffle=True,
    seed=42,
)

print(train_x, train_y)
print(val_x, val_y)
print(test_x, test_y)
```

### Arbitrary Data Splitting
Sometimes you need to create more than 3 splits. For this we provide
[`split_data()`][amltk.sklearn.split_data], which has an identical function
signature, except the `splits=` you specify is a dictionary from the name
of the split to the percentage you wish.

```python exec="true" source="material-block" result="python" title="Arbitrary Data Splitting"
from amltk.sklearn import split_data

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
splits = split_data(x, y, splits={"train": 0.5, "val": 0.3, "test": 0.2}, seed=42)

train_x, train_y = splits["train"]
val_x, val_y = splits["val"]
test_x, test_y = splits["test"]

print(train_x, train_y)
print(val_x, val_y)
print(test_x, test_y)
```
