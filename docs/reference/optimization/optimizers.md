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

For a reference on implementing an optimizer you can refer to any of the following:


## SMAC

::: amltk.optimization.optimizers.smac
    options:
        members: false

## NePs

::: amltk.optimization.optimizers.neps
    options:
        members: false

## Optuna

::: amltk.optimization.optimizers.optuna
    options:
        members: false

## Integrating your own

::: amltk.optimization.optimizer
    options:
        members: false
