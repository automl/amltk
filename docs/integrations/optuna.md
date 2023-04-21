# Optuna
[Optuna](https://optuna.org/) is an automatic hyperparameter optimization
software framework, particularly designed for machine learning.

We provide the follow integrations for Optuna:

* A [`OptunaSpaceAdapter`][byop.optuna.OptunaSpaceAdapter] for [parsing](#parsing-spaces)
an Optuna search space and for [sampling](#sampling)
from an Optuna search space.

??? example "SpaceAdapter Interface"

    This is an implementation of the
    [`SpaceAdapter`][byop.pipeline.space.SpaceAdapter] interface which
    can be used for parsing or sampling anything in AutoML-Toolkit.

* An [`OptunaOptimizer`][byop.optuna.OptunaOptimizer] for optimizing
some given function. See the [Optimizer](#Optimizer)

??? example "Optimizer Interface"

    This is an implementation of the [`Optimizer`][byop.optimization.Optimizer]
    interface which offers an _ask-and-tell_ interface to some underlying optimizer.


## Parser
In general, you should consult the [Optuna documentation](httpsTODO).

You can encode the following things:

```python exec="true" source="material-block" result="python" title="A simple space"
from optuna.distributions import FloatDistribution

from byop.optuna import OptunaSpaceAdapter

search_space = {
    "a": (1, 10),  # An int
    "b": (2.5, 10.0),  # A float
    "c": ["apple", "banana", "carrot"], # A categorical
    "d": FloatDistribution(1e-5, 1, log=True), # An Optuna log float distribution
}

adapter = OptunaSpaceAdapter()

space = adapter.parse(search_space)
print(space)
```

In general, any of these simple types or anything inheriting from
`optuna.BaseDistribution` can be used.

This [`parse()`][byop.pipeline.Parser.parse] method is also
able to parse more complicated objects, such as a [`Step`][byop.pipeline.Step]
or even entire [`Pipelines`][byop.pipeline.Pipeline].

```python exec="true" source="material-block" result="python" title="Parsing Steps"
from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import step

my_step = step(
    "mystep",
    item=object(),
    space={"a": (1, 10), "b": (2.0, 3.0), "c": ["cat", "dog"]}
)

adapter = OptunaSpaceAdapter()
space = adapter.parse(my_step)

print(space)
```

```python exec="true" source="material-block" result="python" title="Parsing a Pipeline"
from optuna.distributions import FloatDistribution

from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import step, Pipeline

my_pipeline = Pipeline.create(
    step("A", item=object(), space={"C": (0.0, 1.0), "initial": (1, 10)}),
    step("B", item=object(), space={"lr": FloatDistribution(1e-5, 1, log=True)}),
)

adapter = OptunaSpaceAdapter()
space = adapter.parse(my_pipeline)

print(space)
```

## Sampler
As [OptunaSpaceAdapter][byop.optuna.OptunaSpaceAdapter] implements the
[Sampler][byop.pipeline.Sampler] interface, you can also [`sample()`][byop.pipeline.Sampler.sample]
from these spaces.

```python exec="true" source="material-block" result="python" title="Sampling from a space"
from byop.optuna import OptunaSpaceAdapter

search_space = {
    "a": (1, 10),
    "b": (0.5, 9.0),
    "c": ["apple", "banana", "carrot"],
}

adapter = OptunaSpaceAdapter()
space = adapter.parse(search_space)
sample = adapter.sample(space)

print(sample)
```

### For use with Step, Pipeline
The [`Pipeline`][byop.pipeline.Pipeline] and [`Step`][byop.pipeline.Step] objects
have a [`space()`][byop.pipeline.Pipeline.space] and
[`sample()`][byop.pipeline.Pipeline.sample] method.
These accept a [`Parser`][byop.pipeline.Parser] and a [`Sampler`][byop.pipeline.Sampler]
interface, for which [`OptunaSpaceAdapter`][byop.optuna.OptunaSpaceAdapter]
supports poth of these interfaces.

```python exec="true" source="material-block" result="python" title="Using Optuna with a Step"
from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import step

my_step = step(
    "mystep",
    item=object(),
    space={"a": (1, 10), "b": (2.0, 3.0), "c": ["cat", "dog"]}
)

space = my_step.space(parser=OptunaSpaceAdapter)
print(space)

sample = my_step.sample(space, sampler=OptunaSpaceAdapter)
print(sample)
```

```python exec="true" source="material-block" result="python" title="Using Optuna with a Pipeline"
from optuna.distributions import FloatDistribution

from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import step, Pipeline

my_pipeline = Pipeline.create(
    step("A", item=object(), space={"C": (0.0, 1.0), "initial": [1, 10]}),
    step("B", item=object(), space={"lr": FloatDistribution(1e-5, 1, log=True)}),
)

space = my_pipeline.space(parser=OptunaSpaceAdapter)
print(space)

sample = my_pipeline.sample(space, sampler=OptunaSpaceAdapter)
print(sample)
```

### For use with RandomSearch
The [`RandomSearch`][byop.optimization.RandomSearch] object accepts a
[`Sampler`][byop.pipeline.Sampler] interface, for which
[`OptunaSpaceAdapter`][byop.optuna.OptunaSpaceAdapter] supports.

```python exec="true" source="material-block" result="python" title="Using Optuna with RandomSearch"
from optuna.distributions import FloatDistribution

from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import step, Pipeline
from byop.optimization import RandomSearch

my_pipeline = Pipeline.create(
    step("A", item=object(), space={"C": (0.0, 1.0), "initial": [1, 10]}),
    step("B", item=object(), space={"lr": FloatDistribution(1e-5, 1, log=True)}),
)
space = my_pipeline.space(parser=OptunaSpaceAdapter)

random_search_optimizer = RandomSearch(
    space=space,
    sampler=OptunaSpaceAdapter,
    seed=10
)

for i in range(3):
    trial = random_search_optimizer.ask()
    print(trial)

    with trial.begin():
        # Run experiment here
        pass

    report = trial.success(cost=1)
    print(report)

    random_search_optimizer.tell(report)
```

## Optimizer
We also integrate Optuna using the [`Optimizer`][byop.optimization.Optimizer] interface.
This requires us to support two keys methods, [`ask()`][byop.optimization.Optimizer.ask]
and [`tell()`][byop.optimization.Optimizer.tell].

```python exec="true" source="material-block" result="python" title="Using Optuna with Optimizer"
from byop.optuna import OptunaOptimizer, OptunaSpaceAdapter
from byop.pipeline import step

item = step(
    "mystep",
    item=object(),
    space={"a": (1, 10), "b": (2.0, 3.0), "c": ["cat", "dog"]}
)

space = item.space(parser=OptunaSpaceAdapter)

# You can forward **kwargs here to `optuna.create_study()`
optimizer = OptunaOptimizer.create(space=space)

for i in range(3):
    trial = optimizer.ask()
    print(trial)

    with trial.begin():
        # Run experiment here
        pass

    report = trial.success(cost=1)
    print(report)

    optimizer.tell(report)
```
