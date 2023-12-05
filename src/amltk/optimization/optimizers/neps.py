"""The [`NEPSOptimizer`][amltk.optimization.optimizers.neps.NEPSOptimizer],
is a wrapper around the [`NePs`](https://github.com/automl/neps) optimizer.

!!! tip "Requirements"

    This requires `smac` which can be installed with:

    ```bash
    pip install amltk[neps]

    # Or directly
    pip install neural-pipeline-search
    ```

!!! warning "NePs is still in development"

    NePs is still in development and is not yet stable.
    There are likely going to be issues. Please report any issues to NePs or in
    AMLTK.

This uses `ConfigSpace` as its [`search_space()`][amltk.pipeline.Node.search_space] to
optimize.

Users should report results using
[`trial.success(loss=...)`][amltk.optimization.Trial.success]
where `loss=` is a scaler value to **minimize**. Optionally,
you can also return a `cost=` which is used for more budget aware algorithms.
Again, please see NeP's documentation for more.

!!! warning "Conditionals in ConfigSpace"

    NePs does not support conditionals in its search space. This is account
    for when using the
    [`preferred_parser()`][amltk.optimization.optimizers.neps.NEPSOptimizer.preferred_parser].
    during search space creation. In this case, it will simply remove all conditionals
    from the search space, which may not be ideal for the given problem at hand.

Visit their documentation for what you can pass to
[`NEPSOptimizer.create()`][amltk.optimization.optimizers.neps.NEPSOptimizer.create].

The below example shows how you can use neps to optimize an sklearn pipeline.

```python
from __future__ import annotations

import logging

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from amltk.optimization.optimizers.neps import NEPSOptimizer
from amltk.scheduling import Scheduler
from amltk.optimization import History, Trial, Metric
from amltk.pipeline import Component

logging.basicConfig(level=logging.INFO)


def target_function(trial: Trial, pipeline: Pipeline) -> Trial.Report:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = pipeline.configure(trial.config).build("sklearn")

    with trial.begin():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        loss = 1 - accuracy
        return trial.success(loss=loss, accuracy=accuracy)

    return trial.fail()
from amltk._doc import make_picklable; make_picklable(target_function)  # markdown-exec: hide

pipeline = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})

metric = Metric("accuracy", minimize=False, bounds=(0, 1))
optimizer = NEPSOptimizer.create(space=pipeline, metrics=metric, bucket="neps-doc-example")

N_WORKERS = 2
scheduler = Scheduler.with_processes(N_WORKERS)
task = scheduler.task(target_function)

history = History()

@scheduler.on_start(repeat=N_WORKERS)
def on_start():
    trial = optimizer.ask()
    task.submit(trial, pipeline)

@task.on_result
def tell_and_launch_trial(_, report: Trial.Report):
    if scheduler.running():
        optimizer.tell(report)
        trial = optimizer.ask()
        task.submit(trial, pipeline)

@task.on_result
def add_to_history(_, report: Trial.Report):
    history.add(report)

scheduler.run(timeout=3, wait=False)

print(history.df())
optimizer.bucket.rmdir()  # markdown-exec: hide
```

!!! todo "Deep Learning"

    Write an example demonstrating NEPS with continuations

!!! todo "Graph Search Spaces"

    Write an example demonstrating NEPS with its graph search spaces

"""  # noqa: E501
from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from typing_extensions import override

import metahyper.api
from ConfigSpace import ConfigurationSpace
from metahyper import instance_from_map
from more_itertools import first_true
from neps.optimizers import SearcherMapping
from neps.search_spaces.parameter import Parameter
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace

import amltk.randomness
from amltk.optimization import Optimizer, Trial
from amltk.pipeline import Node
from amltk.pipeline.parsers.configspace import parser as configspace_parser
from amltk.store import PathBucket

if TYPE_CHECKING:
    from typing_extensions import Self

    from neps.api import BaseOptimizer

    from amltk.optimization.metric import Metric
    from amltk.types import Seed

    class NEPSPreferredParser(Protocol):
        """The preferred parser call signature for NEPSOptimizer."""

        def __call__(
            self,
            node: Node,
            *,
            seed: int | None = None,
            flat: bool = False,
            delim: str = ":",
        ) -> ConfigurationSpace:
            """See [`configspace_parser`][amltk.pipeline.parsers.configspace.parser]."""
            ...


logger = logging.getLogger(__name__)


@dataclass
class NEPSTrialInfo:
    """The info for a trial."""

    name: str
    config: dict[str, Any]
    pipeline_directory: Path
    previous_pipeline_directory: Path | None


def _to_neps_space(
    space: SearchSpace
    | ConfigurationSpace
    | Mapping[str, ConfigurationSpace | Parameter],
) -> SearchSpace:
    if isinstance(space, SearchSpace):
        return space

    try:
        # Support pipeline space as ConfigurationSpace definition
        if isinstance(space, ConfigurationSpace):
            space = pipeline_space_from_configspace(space)

        # Support pipeline space as mix of ConfigurationSpace and neps parameters
        new_pipeline_space: dict[str, Parameter] = {}
        for key, value in space.items():
            if isinstance(value, ConfigurationSpace):
                config_space_parameters = pipeline_space_from_configspace(value)
                new_pipeline_space = {**new_pipeline_space, **config_space_parameters}
            else:
                new_pipeline_space[key] = value
        space = new_pipeline_space

        # Transform to neps internal representation of the pipeline space
        return SearchSpace(**space)
    except TypeError as e:
        message = f"The pipeline_space has invalid type: {type(space)}"
        raise TypeError(message) from e


def _to_neps_searcher(
    *,
    space: SearchSpace,
    searcher: str | BaseOptimizer | None = None,
    loss_value_on_error: float | None = None,
    cost_value_on_error: float | None = None,
    max_cost_total: float | None = None,
    ignore_errors: bool = False,
    searcher_kwargs: Mapping[str, Any] | None = None,
) -> BaseOptimizer:
    if searcher == "default" or searcher is None:
        if space.has_fidelity:
            searcher = "hyperband"
            if hasattr(space, "has_prior") and space.has_prior:
                searcher = "hyperband_custom_default"
        else:
            searcher = "bayesian_optimization"
        logger.info(f"Running {searcher} as the searcher")

    _searcher_kwargs = dict(searcher_kwargs) if searcher_kwargs else {}
    _searcher_kwargs.update(
        {
            "loss_value_on_error": loss_value_on_error,
            "cost_value_on_error": cost_value_on_error,
            "ignore_errors": ignore_errors,
        },
    )
    return instance_from_map(SearcherMapping, searcher, "searcher", as_class=True)(
        pipeline_space=space,
        budget=max_cost_total,  # TODO: use max_cost_total everywhere
        **_searcher_kwargs,
    )


class NEPSOptimizer(Optimizer[NEPSTrialInfo]):
    """An optimizer that uses SMAC to optimize a config space."""

    def __init__(
        self,
        *,
        space: SearchSpace,
        loss_metric: Metric,
        cost_metric: Metric | None = None,
        optimizer: BaseOptimizer,
        working_dir: Path,
        seed: Seed | None = None,
        bucket: PathBucket | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            space: The space to use.
            loss_metric: The metric to optimize.
            cost_metric: The cost metric to use. Only certain NePs optimizers support
            optimizer: The optimizer to use.
            seed: The seed to use for the trials (and not optimizers).
            working_dir: The directory to use for the trials.
            bucket: The bucket to give to trials generated from this optimizer.
        """
        if isinstance(loss_metric, Sequence):
            raise ValueError("NePs does not support multiple metrics")

        if cost_metric is not None and cost_metric.minimize is False:
            raise ValueError("NePs only supports minimizing cost metrics")

        if cost_metric is None and optimizer.budget is not None:
            raise ValueError(
                "NePs optimizers with a budget require a cost metric to be provided",
            )

        metrics = [loss_metric]
        if cost_metric is not None:
            metrics.append(cost_metric)

        super().__init__(bucket=bucket, metrics=metrics)
        self.space = space
        self.seed = amltk.randomness.as_int(seed)
        self.optimizer = optimizer
        self.working_dir = working_dir
        self.loss_metric = loss_metric
        self.cost_metric = cost_metric

        self.optimizer_state_file = self.working_dir / "optimizer_state.yaml"
        self.base_result_directory = self.working_dir / "results"
        self.serializer = metahyper.utils.YamlSerializer(self.optimizer.load_config)

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.base_result_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        *,
        space: (
            SearchSpace
            | ConfigurationSpace
            | Mapping[str, ConfigurationSpace | Parameter]
            | Node
        ),
        metrics: Metric,
        cost_metric: Metric | None = None,
        bucket: PathBucket | str | Path | None = None,
        searcher: str | BaseOptimizer = "default",
        working_dir: str | Path = "neps",
        overwrite: bool = True,
        seed: Seed | None = None,
        max_cost_total: float | None = None,
        searcher_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Create a new NEPS optimizer.

        Args:
            space: The space to use.
            metrics: The metrics to optimize.

                !!! warning

                    NePs does not support multiple metrics. Please only pass a single
                    metric.

            cost_metric: The cost metric to use. Only certain NePs optimizers support
                this.
            seed: The seed to use for the trials.

                !!! warning

                    NePS optimizers do not support an explicit seeding. If you'd
                    like to seed their optimizers, they use the global
                    `torch.manual_seed`, `np.random.seed`, and `random.seed`.
                    This is not considered a good practice and there is not
                    much we can do from AMLTK to help with this.

            bucket: The bucket to give to trials generated by this optimizer.
            searcher: The searcher to use.
            working_dir: The directory to use for the optimization.
            overwrite: Whether to overwrite the working directory if it exists.
            max_cost_total: The maximum cost to use for the optimization.

                !!! warning

                    This only effects the optimization if the searcher utilizes the
                    budget for it's actual suggestion of the next config. If the
                    searcher does not use the budget. This parameter has no effect.

                    The user is still expected to stop `ask()`'ing for configs when
                    they have reached some budget.
            searcher_kwargs: Additional kwargs to pass to the searcher.
        """
        if isinstance(space, Node):
            space = space.search_space(parser=NEPSOptimizer.preferred_parser())

        match bucket:
            case None:
                bucket = PathBucket(
                    f"{cls.__name__}-{datetime.now().isoformat()}",
                )
            case str() | Path():
                bucket = PathBucket(bucket)
            case bucket:
                bucket = bucket  # noqa: PLW0127

        space = _to_neps_space(space)
        searcher = _to_neps_searcher(
            space=space,
            searcher=searcher,
            max_cost_total=max_cost_total,
            searcher_kwargs=searcher_kwargs,
        )
        working_dir = Path(working_dir)
        if working_dir.exists() and overwrite:
            logger.info(f"Removing existing working directory {working_dir}")
            shutil.rmtree(working_dir)

        return cls(
            space=space,
            bucket=bucket,
            seed=seed,
            loss_metric=metrics,
            cost_metric=cost_metric,
            optimizer=searcher,
            working_dir=working_dir,
        )

    @override
    def ask(self) -> Trial[NEPSTrialInfo]:
        """Ask the optimizer for a new config.

        Returns:
            The trial info for the new config.
        """
        with self.optimizer.using_state(self.optimizer_state_file, self.serializer):
            (
                config_id,
                config,
                pipeline_directory,
                previous_pipeline_directory,
            ) = metahyper.api._sample_config(  # type: ignore
                optimization_dir=self.working_dir,
                sampler=self.optimizer,
                serializer=self.serializer,
                logger=logger,
            )

        match config:
            case SearchSpace():
                _config = config.hp_values()
            case _:  # type: ignore
                _config = {
                    k: v.value if isinstance(v, Parameter) else v
                    for k, v in config.items()  # type: ignore
                }

        info = NEPSTrialInfo(
            name=str(config_id),
            config=deepcopy(_config),
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
        )

        match self.cost_metric:
            case None:
                metrics = [self.loss_metric]
            case cost_metric:
                metrics = [self.loss_metric, cost_metric]

        trial = Trial(
            name=info.name,
            config=info.config,
            info=info,
            seed=self.seed,
            bucket=self.bucket,
            metrics=metrics,
        )
        logger.debug(f"Asked for trial {trial.name}")
        return trial

    @override
    def tell(self, report: Trial.Report[NEPSTrialInfo]) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            report: The report of the trial.
        """
        logger.debug(f"Telling report for trial {report.trial.name}")
        info = report.info
        assert info is not None

        # Get a metric result
        metric_result = first_true(
            report.metric_values,
            pred=lambda value: value.metric.name == self.loss_metric.name,
            default=self.loss_metric.worst,
        )

        # Convert metric result to a minimization loss
        neps_loss: float
        if (_loss := metric_result.distance_to_optimal) is not None:
            neps_loss = _loss
        else:
            neps_loss = metric_result.loss

        result: dict[str, Any] = {"loss": neps_loss}
        metadata: dict[str, Any] = {"time_end": report.time.end}

        if self.cost_metric is not None:
            cost_metric: Metric = self.cost_metric
            _cost = first_true(
                report.metric_values,
                pred=lambda value: value.metric.name == cost_metric.name,
                default=self.cost_metric.worst,
            )
            cost = _cost.value
            result["cost"] = cost

            # If it's a budget aware optimizer
            if self.optimizer.budget is not None:
                with self.optimizer.using_state(
                    self.optimizer_state_file,
                    self.serializer,
                ):
                    self.optimizer.used_budget += cost

                metadata["budget"] = {
                    "max": self.optimizer.budget,
                    "used": self.optimizer.used_budget,
                    "eval_cost": cost,
                    "account_for_cost": True,
                }

        # Dump results
        self.serializer.dump(result, info.pipeline_directory / "result")

        # Load and dump metadata
        config_metadata = self.serializer.load(info.pipeline_directory / "metadata")
        config_metadata.update(metadata)
        self.serializer.dump(config_metadata, info.pipeline_directory / "metadata")

    @override
    @classmethod
    def preferred_parser(cls) -> NEPSPreferredParser:
        """The preferred parser for this optimizer."""
        # TODO: We might want a custom one for neps.SearchSpace, for now we will
        # use config space but without conditions as NePs doesn't support conditionals
        return partial(configspace_parser, conditionals=False)
