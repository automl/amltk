## Search Spaces
* [ConfigSpace](./configspace) - A serializable search space definition which
 supports choices and search space constraints. A great default go to!
* [Optuna](./optuna) - A search space for using Optuna as your optimizer. Only
 done as a static definition and currently does not support Optuna's define-by-run.

## Optimizers

* [SMAC](./smac) - A powerful Bayesian-Optimization framework, primarly based on a custom
 Random Forest, supporting complex conditionals in a bayesian manner.
* [Optuna](./optuna) - A highly flexible Optimization framework based on Tree-Parzan
 Estimators.

## Pipeline Builders

* [sklearn](./sklearn) - Export your pipelines to a pure [sklearn.pipeline.Pipeline][]
    and some utility to ease data splitting.

## Scheduler Executors

* [DaskJobQueue](./dask-jobqueue.md) - A set of [`Executors`][concurrent.futures.Executor]
    usable with the [`Scheduler`][amltk.Scheduler] for different cluster setups.

## Plugins

* [CallLimiter](./call_limiter.md) - A simple plugin to limit how many times your task
    can be called, how many concurrent instances of it can be run and prevent a task being
    submitted while another task is running.
* [pynisher](./pynisher.md) - A plugin to limit the maximum time or memory a task can
    use, highly suitable for creating AutoML systems.
* [wandb](./wandb.md) - A plugin that automatically logs your runs to
    [weights and biases](https://wandb.ai/site)!
* [threadpoolctl](./threadpoolctl.md) - A plugin that uses
[`threadpoolctl`](https://github.com/joblib/threadpoolctl) to limit the number of threads used
by certain numerical libraries within a tasks execution.

## Utility

* [Buckets](./buckets.md) - A nice utility to view the file system in a dictionary like
    fashion, enabling quick and easy storing of many file types at once.
* [History](./history.md) - A datastructure to house the results of an optimization run and
    pull out information after.
* [data](./data.md) - Utilities for working with data containers like numpy arrays, pandas
    dataframes and series.
