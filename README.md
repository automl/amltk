# byop
A set of re-usable components for building complex
configurable pipelines, agnostic to:
* Search space implementation
* How to build your pipeline
* Where compute happens

... yet providing sensible defaults and options to plug in your own.


# Examples
In progress

### Defining, Configuring and building a Pipeline
```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from byop import Pipeline, choice, split, step

# Some toy data
X = pd.DataFrame({"a": ["1", "0", "1", "dog"], "b": [4, 5, 6, 7], "c": [7, 8, 9, 10]})
y = pd.Series([1, 0, 1, 1])

# Defining a pipeline which splits data, applies
# different preprocessing to each, finally applying a choice
# between a RandomForest or a SVM
pipeline = Pipeline.create(
    split(
        "feature_preprocessing",
        step("cats", OrdinalEncoder) | step("std", StandardScaler),
        step("nums", StandardScaler),
        item=ColumnTransformer,
        config={
            "cats": make_column_selector(dtype_include=object),
            "nums": make_column_selector(dtype_include=np.number),
        },
    ),
    choice(
        "algorithm",
        step(
            "rf",
            item=RandomForestClassifier,
            space={
                "n_estimators": [10, 100],
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),
        step("svm", SVC, config={"C": [0.1, 1, 10]}),
    ),
)

# If ConfigSpace is installed, this will spit out
# a configspace
space = pipeline.space(seed=1)
config = space.sample_configuration()

# Configure your pipeline down to what was chosen
configured_pipeline = pipeline.configure(config)

# Take your configured pipeline and give back a pure
# sklearn pipeline, automatically deciding so based
# on the `item`s present
sklearn_pipeline = configured_pipeline.build()
assert isinstance(sklearn_pipeline, SklearnPipeline)

# This is all normal sklearn stuff
sklearn_pipeline = sklearn_pipeline.fit(X, y)
sklearn_pipeline.predict(X)
```

### Scheduler
TODO:


## Installation
```bash
git clone git@github:automl/byop.git
pip install -e ".[dev]"
```

## Docs
This library uses [`mkdocs`](https://squidfunk.github.io/mkdocs-material/getting-started/) for markdown style documentation.
```bash
just docs
# Click link given
```

The configuration can mainly be found `mkdocs.yml` with
the navigation in `mkdocs-nav.yml`.

## Code Quality
This library uses [`ruff`](https://github.com/charliermarsh/ruff) and [`black`](https://github.com/psf/black)
for code quality checks. You can run these manually or use the following
`just` commands.

```python
just fix  # Some automated fixes
just check
```

Their configuration is `pyproject.toml`


## Tests
```bash
pytest
```

## Versioning
Uses [conventional-commits](https://www.conventionalcommits.org/en/v1.0.0/#summary). Will create
a new version based on commit messages, updating the changelog, creating a tag
and finally **pushing** the current branch and version tag to github.
```bash
just bump
```
