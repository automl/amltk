## Optimizers
An [`Optimizer`][amltk.optimization.Optimizer]'s goal is to achieve the optimal
value for a given [`Metric`][amltk.optimization.Metric] or `Metrics` using
repeated [`Trials`][amltk.optimization.Trial].

What differentiates AMLTK from other optimization libraries is that we rely solely
on optimizers that support an _"Ask-and-Tell"_ interface.
This means we can _"Ask"_ and optimizer for its next suggested [`Trial`][amltk.optimization.Trial],
and we can _"Tell"_ it a [`Report`][amltk.optimization.Trial.Report] when we have one.
**In fact, here's the required interface.**

```python
class Optimizer:

    def tell(self, report: Trial.Report) -> None: ...

    def ask(self) -> Trial: ...
```

Now we do require optimizers to implement these `ask()` and `tell()` methods, correctly filling
in a [`Trial`][amltk.optimization.Trial] with appropriate parsing out results from
the [`Report`][amltk.optimization.Trial.Report], as this will be different for every optimizer.

??? note "Why only Ask and Tell Optimizers?"

    1. **Easy Parallelization**: Many optimizers handle running the function to optimize and hence
        roll out their own parallelization schemes and store data in all various different ways. By taking
        this repsonsiblity away from an optimzer and giving it to the user, we can easily parallelize how
        we wish

    2. **API maintenance**: Many optimziers are research code and hence a bit unstable with resepct to their
        API so wrapping around them can be difficult. By requiring this _"Ask-and-Tell"_ interface,
        we reduce the complexity of what is required of both the "Optimizer" and wrapping it.

    3. **Full Integration**: We can fully hook into the life cycle of a running optimizer. We are not relying
        on the optimizer to support callbacks at every step of their _hot-loop_ and as such, we
        can fully leverage all the other systems of AutoML-toolkit

    4. **Easy Integration**: it makes developing and integrating new optimizers easy. You only have
        to worry that the internal state of the optimizer is updated accordingly to these
        two _"Ask"_ and _"Tell"_ events and that's it.

For a reference on implementing an optimizer you can refer to any of the following
API Docs:
* [SMAC][amltk.optimization.optimizers.smac]
* [NePs][amltk.optimization.optimizers.neps]
* [Optuna][amltk.optimization.optimizers.optuna]
* [Random Search][amltk.optimization.optimizers.random_search]

## Integrating your own
The base [`Optimizer`][amltk.optimization.optimizer.Optimizer] class,
defines the API we require optimizers to implement.

* [`ask()`][amltk.optimization.optimizer.Optimizer.ask] - Ask the optimizer for a
    new [`Trial`][amltk.optimization.trial.Trial] to evaluate.
* [`tell()`][amltk.optimization.optimizer.Optimizer.tell] - Tell the optimizer
    the result of the sampled config. This comes in the form of a
    [`Trial.Report`][amltk.optimization.trial.Trial.Report].

Additionally, to aid users from switching between optimizers, the
[`preferred_parser()`][amltk.optimization.optimizer.Optimizer.preferred_parser]
method should return either a `parser` function or a string that can be used
with [`node.search_space(parser=..._)`][amltk.pipeline.Node.search_space] to
extract the search space for the optimizer.

Please refer to the code of [Random Search][amltk.optimization.optimizers.random_search]
on github for an example of how to implement a new optimizer.
