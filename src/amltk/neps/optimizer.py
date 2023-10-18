"""A thin wrapper around NEPS to make it easier to use with AutoMLToolkit.

TODO: More description and explanation with examples.
"""
from __future__ import annotations

import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping
from typing_extensions import override

import metahyper.api
from ConfigSpace import ConfigurationSpace
from metahyper import instance_from_map
from neps.optimizers import SearcherMapping
from neps.search_spaces.parameter import Parameter
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace

from amltk.optimization import Optimizer, Trial

if TYPE_CHECKING:
    from typing_extensions import Self

    from neps.api import BaseOptimizer


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
        optimizer: BaseOptimizer,
        working_dir: Path,
        ignore_errors: bool = True,
        loss_value_on_error: float | None = None,
        cost_value_on_error: float | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            space: The space to use.
            optimizer: The optimizer to use.
            working_dir: The directory to use for the optimization.
            ignore_errors: Whether the optimizers should ignore errors from trials.
            loss_value_on_error: The value to use for the loss if the trial fails.
            cost_value_on_error: The value to use for the cost if the trial fails.
        """
        super().__init__()
        self.space = space
        self.optimizer = optimizer
        self.working_dir = working_dir
        self.ignore_errors = ignore_errors
        self.loss_value_on_error = loss_value_on_error
        self.cost_value_on_error = cost_value_on_error

        self.optimizer_state_file = self.working_dir / "optimizer_state.yaml"
        self.base_result_directory = self.working_dir / "results"
        self.serializer = metahyper.utils.YamlSerializer(self.optimizer.load_config)

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.base_result_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        *,
        space: (
            SearchSpace
            | ConfigurationSpace
            | Mapping[str, ConfigurationSpace | Parameter]
        ),
        searcher: str | BaseOptimizer = "default",
        working_dir: str | Path = "neps",
        overwrite: bool = False,
        loss_value_on_error: float | None = None,
        cost_value_on_error: float | None = None,
        max_cost_total: float | None = None,
        ignore_errors: bool = True,
        searcher_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Create a new NEPS optimizer.

        Args:
            space: The space to use.
            searcher: The searcher to use.
            working_dir: The directory to use for the optimization.
            overwrite: Whether to overwrite the working directory if it exists.
            loss_value_on_error: The value to use for the loss if the trial fails.
            cost_value_on_error: The value to use for the cost if the trial fails.
            max_cost_total: The maximum cost to use for the optimization.

                !!! warning

                    This only effects the optimization if the searcher utilizes the
                    budget for it's actual suggestion of the next config. If the
                    searcher does not use the budget. This parameter has no effect.

                    The user is still expected to stop `ask()`'ing for configs when
                    they have reached some budget.

            ignore_errors: Whether the optimizers should ignore errors from trials
                or whether they should be taken into account. Please set `loss_on_value`
                and/or `cost_value_on_error` if you set this to `False`.
            searcher_kwargs: Additional kwargs to pass to the searcher.
        """
        space = _to_neps_space(space)
        searcher = _to_neps_searcher(
            space=space,
            searcher=searcher,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            max_cost_total=max_cost_total,
            ignore_errors=ignore_errors,
            searcher_kwargs=searcher_kwargs,
        )
        working_dir = Path(working_dir)
        if working_dir.exists() and overwrite:
            logger.info(f"Removing existing working directory {working_dir}")
            shutil.rmtree(working_dir)

        return cls(
            space=space,
            optimizer=searcher,
            working_dir=working_dir,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
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
                _,
            ) = metahyper.api._sample_config(  # type: ignore
                optimization_dir=self.working_dir,
                sampler=self.optimizer,
                serializer=self.serializer,
                logger=logger,
            )

        if isinstance(config, SearchSpace):
            _config = config.hp_values()
        else:
            _config = {
                k: v.value if isinstance(v, Parameter) else v for k, v in config.items()
            }

        info = NEPSTrialInfo(
            name=str(config_id),
            config=deepcopy(_config),
            pipeline_directory=pipeline_directory,
            previous_pipeline_directory=previous_pipeline_directory,
        )
        trial = Trial(name=info.name, config=info.config, info=info, seed=None)
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

        # This is how NEPS handles errors
        result: Literal["error"] | dict[str, Any]
        if report.status in (Trial.Status.CRASHED, Trial.Status.FAIL):
            result = "error"
        else:
            result = report.results

        metadata: dict[str, Any] = {"time_end": report.time.end}
        if result == "error":
            if not self.ignore_errors:
                if self.loss_value_on_error is not None:
                    report.results["loss"] = self.loss_value_on_error
                if self.cost_value_on_error is not None:
                    report.results["cost"] = self.cost_value_on_error
        else:
            if (loss := result.get("loss")) is not None:
                report.results["loss"] = float(loss)
            else:
                raise ValueError(
                    "The 'loss' should be provided if the trial is successful"
                    f"\n{result=}",
                )

            cost = result.get("cost")
            if (cost := result.get("cost")) is not None:
                cost = float(cost)
                result["cost"] = cost
                account_for_cost = result.get("account_for_cost", True)

                if account_for_cost:
                    with self.optimizer.using_state(
                        self.optimizer_state_file,
                        self.serializer,
                    ):
                        self.optimizer.used_budget += cost

                metadata["budget"] = {
                    "max": self.optimizer.budget,
                    "used": self.optimizer.used_budget,
                    "eval_cost": cost,
                    "account_for_cost": account_for_cost,
                }
            elif self.optimizer.budget is not None:
                raise ValueError(
                    "'cost' should be provided when the optimizer has a budget"
                    f"\n{result=}",
                )

        # Dump results
        self.serializer.dump(result, info.pipeline_directory / "result")

        # Load and dump metadata
        config_metadata = self.serializer.load(info.pipeline_directory / "metadata")
        config_metadata.update(metadata)
        self.serializer.dump(config_metadata, info.pipeline_directory / "metadata")
