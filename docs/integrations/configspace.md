# ConfigSpace
[ConfigSpace](https://automl.github.io/ConfigSpace/master/) is a library for
representing and sampling configurations for hyperparameter optimization.
It features a straightforward API for defining hyperparameters, their ranges
and even conditional dependencies.

It is generally flexible enough for more complex use cases, even
handling the complex pipelines of [AutoSklearn](https://automl.github.io/auto-sklearn/master/).
and [AutoPyTorch](https://automl.github.io/Auto-PyTorch/master/), large
scale hyperparameter spaces over which to optimize entire
pipelines at a time.

We integrate [ConfigSpace](https://automl.github.io/ConfigSpace/master/) with
AutoML-Toolkit by allowing you to parse out entire spaces
from a [Pipeline][byop.pipeline.Pipeline] and sample from
these spaces.

Check out the [API doc][byop.configspace.ConfigSpaceAdapter] for more info.

!!! note "Space Adapter Interface"

    This integration is provided by implementing the
    [SpaceAdapater][byop.pipeline.SpaceAdapter] interface.
    Check out its documentation for implementing your own.


## Parsing Spaces
In general, you should consult the
[ConfigSpace documentation](https://automl.github.io/ConfigSpace/master/).
Anything you can insert into a `ConfigurationSpace` object is valid.

Here's an example of a simple space using pure python objects.

```python exec="true" source="material-block" result="python" title="A simple space"
from byop.configspace import ConfigSpaceAdapter

search_space = {
    "a": (1, 10),
    "b": (0.5, 9.0),
    "c": ["apple", "banana", "carrot"],
}

adapter = ConfigSpaceAdapter()
space = adapter.parse(search_space)
print(space)
```

You can specify more complex spaces using the `Integer`, `Float` and
`Categorical` functions from ConfigSpace.

```python exec="true" source="material-block" result="python" title="A more complicated space"
from ConfigSpace import Integer, Float, Categorical, Normal
from byop.configspace import ConfigSpaceAdapter

search_space = {
    "a": Integer("a", bounds=(1, 1000), log=True),
    "b": Float("b", bounds=(2.0, 3.0), distribution=Normal(2.5, 0.1)),
    "c": Categorical("c", ["small", "medium", "large"], ordered=True),
}

adapter = ConfigSpaceAdapter()
space = adapter.parse(search_space)
print(space)
```

Lastly, this [`parse()`][byop.pipeline.Parser.parse] method is also
able to parse more complicated objects, such as a [`Step`][byop.pipeline.Step]
or even entire [`Pipelines`][byop.pipeline.Pipeline].

```python exec="true" source="material-block" result="python" title="Parsing Steps"
from byop.configspace import ConfigSpaceAdapter
from byop.pipeline import step

my_step = step(
    "mystep",
    item=object(),
    space={"a": (1, 10), "b": (2.0, 3.0), "c": ["cat", "dog"]}
)

adapter = ConfigSpaceAdapter()
space = adapter.parse(my_step)

print(space)
```

```python exec="true" source="material-block" result="python" title="Parsing a Pipeline"
from ConfigSpace import Float

from byop.configspace import ConfigSpaceAdapter
from byop.pipeline import step, choice, Pipeline

my_pipeline = Pipeline.create(
    choice(
        "algorithm",
        step("A", item=object(), space={"C": (0.0, 1.0), "initial": [1, 10]}),
        step("B", item=object(), space={"lr": Float("lr", (1e-5, 1), log=True)}),
    )
)

adapter = ConfigSpaceAdapter()
space = adapter.parse(my_pipeline)

print(space)
```

## Sampling Spaces
As [ConfigSpaceAdapter][byop.configspace.ConfigSpaceAdapter] implements the
[Sampler][byop.pipeline.Sampler] interface, you can also [`sample()`][byop.pipeline.Sampler.sample]
from these spaces.

```python exec="true" source="material-block" result="python" title="Sampling from a space"
from byop.configspace import ConfigSpaceAdapter

search_space = {
    "a": (1, 10),
    "b": (0.5, 9.0),
    "c": ["apple", "banana", "carrot"],
}

adapter = ConfigSpaceAdapter()
space = adapter.parse(search_space)
sample = adapter.sample(space)

print(sample)
```

### For use with Step, Pipeline
The [`Pipeline`][byop.pipeline.Pipeline] and [`Step`][byop.pipeline.Step] objects
have a [`space()`][byop.pipeline.Pipeline.space] and
[`sample()`][byop.pipeline.Pipeline.sample] method.
These accept a [`Parser`][byop.pipeline.Parser] and a [`Sampler`][byop.pipeline.Sampler]
interface, for which [`ConfigSpaceAdapter`][byop.configspace.ConfigSpaceAdapter]
supports poth of these interfaces.

```python exec="true" source="material-block" result="python" title="Using ConfigSpace with a Step"
from byop.configspace import ConfigSpaceAdapter
from byop.pipeline import step

my_step = step(
    "mystep",
    item=object(),
    space={"a": (1, 10), "b": (2.0, 3.0), "c": ["cat", "dog"]}
)

space = my_step.space(parser=ConfigSpaceAdapter)
print(space)

sample = my_step.sample(space, sampler=ConfigSpaceAdapter)
print(sample)
```

```python exec="true" source="material-block" result="python" title="Using ConfigSpace with a Pipeline"
from ConfigSpace import Float

from byop.configspace import ConfigSpaceAdapter
from byop.pipeline import step, choice, Pipeline

my_pipeline = Pipeline.create(
    choice(
        "algorithm",
        step("A", item=object(), space={"C": (0.0, 1.0), "initial": [1, 10]}),
        step("B", item=object(), space={"lr": Float("lr", (1e-5, 1), log=True)}),
    )
)

space = my_pipeline.space(parser=ConfigSpaceAdapter)
print(space)

sample = my_pipeline.sample(space, sampler=ConfigSpaceAdapter)
print(sample)
```

### For use with RandomSearch
The [`RandomSearch`][byop.optimization.RandomSearch] object accepts a
[`Sampler`][byop.pipeline.Sampler] interface, for which
[`ConfigSpaceAdapter`][byop.configspace.ConfigSpaceAdapter] supports.

```python exec="true" source="material-block" result="python" title="Using ConfigSpace with RandomSearch"
from ConfigSpace import Float

from byop.configspace import ConfigSpaceAdapter
from byop.pipeline import step, choice, Pipeline
from byop.optimization import RandomSearch

my_pipeline = Pipeline.create(
    choice(
        "algorithm",
        step("A", item=object(), space={"C": (0.0, 1.0), "initial": [1, 10]}),
        step("B", item=object(), space={"lr": Float("lr", (1e-5, 1), log=True)}),
    )
)
space = my_pipeline.space(parser=ConfigSpaceAdapter)

random_search_optimizer = RandomSearch(
    space=space,
    sampler=ConfigSpaceAdapter,
    seed=10
)

for i in range(3):
    trial = random_search_optimizer.ask()
    print(trial)
```

