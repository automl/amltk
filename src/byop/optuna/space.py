"""A module to help construct an optuna search space for a pipeline.

``python
from byop.pipeline import Pipeline
from byop.optuna_space import generate_optuna_search_space
 = Pipeline(...)
configspace = generate_optuna_search_space(pipeline)
```
"""
from __future__ import annotations

from typing import Any, Mapping, TypeAlias

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from byop.pipeline import Choice, Pipeline, Searchable, Split, Step

HyperparameterType: TypeAlias = int | str | float
OptunaSearchSpace: TypeAlias = dict[str, BaseDistribution]


def _convert_hp_to_optuna_distribution(
    hp: tuple | list | HyperparameterType, name: str
) -> BaseDistribution:
    if isinstance(hp, tuple):
        if len(hp) != 2:  # noqa: PLR2004
            raise ValueError(f"{name} must be (lower, upper) bound, got {hp}")
        lower, upper = hp
        if type(lower) != type(upper):
            raise ValueError(
                f"Expected {name} to have same type for lower and upper bound,"
                f"got lower: {type(lower)}, upper: {type(upper)}."
            )

        real_hp: BaseDistribution
        if isinstance(lower, float):
            real_hp = FloatDistribution(lower, upper)
        else:
            real_hp = IntDistribution(lower, upper)

    # Lists are categoricals
    elif isinstance(hp, list):
        if len(hp) == 0:
            raise ValueError(f"Can't have empty list for categorical {name}")

        real_hp = CategoricalDistribution(hp)

    # If it's an allowed type, it's a constant
    elif isinstance(hp, HyperparameterType):  # type: ignore[misc, arg-type]
        real_hp = CategoricalDistribution([hp])
    else:
        raise ValueError(
            f"Expected hyperparameter value for {name} to be one of "
            f"tuple | list | int | str | float, got {type(hp)}"
        )
    return real_hp


def _extract_search_space(
    space: Mapping[str, Any],
    prefix: str,
    delimiter: str,
) -> OptunaSearchSpace:
    """Extracts "Define-and-run" search space compatible with
    Optuna study for a given step.

    Args:
        space: search space of the step.
        prefix: prefix that is added to the name of each hyperparameter,
            usually the name of parent steps.
        delimiter: Symbol used to join the prefix with the name of the hyperparameter.

    Returns:
        OptunaSearchSpace
    """
    search_space: OptunaSearchSpace = {}
    for name, hp in space.items():
        if isinstance(hp, BaseDistribution):
            subspace = hp
        else:
            subspace = _convert_hp_to_optuna_distribution(hp=hp, name=name)
        search_space[f"{prefix}{delimiter}{name}"] = subspace
    return search_space


def generate_optuna_search_space(
    pipeline: Pipeline,
) -> OptunaSearchSpace:
    """Generates the search space for the given pipeline.

    Args:
        pipeline: The pipeline to generate the space for.

    Returns:
        OptunaSearchSpace
    """
    # Process main pipeline
    search_space: OptunaSearchSpace = {}
    for splits, _, step in pipeline.walk():
        _splits = splits if splits is not None else []
        subspace = _process_step(_splits, step)
        search_space.update(subspace)

    # Process modules
    for module_name, module in pipeline.modules.items():
        module_space = {
            f"{module_name}:{k}": v
            for k, v in generate_optuna_search_space(module).items()
        }
        search_space.update(module_space)

    # Process searchables
    for searchable_name, searchable in pipeline.searchables.items():
        searchables_space = {
            f"{searchable_name}:{k}": v
            for k, v in _process_step(splits=[], step=searchable).items()
        }
        search_space.update(searchables_space)

    return search_space


def _process_step(
    splits: list[Split],
    step: Step,
    *,
    delimiter: str = ":",
) -> OptunaSearchSpace:
    """Returns the subspace for the given step of the pipeline.

    Args:
        splits (list[Split]): The list of steps in the pipeline
            where the flow of the data is split to reach this step.
        step (Step): step to extract search space from
        delimiter (str): Delimiter used to separate different steps in
            the hyperparameter name. Defaults to ":".

    Raises:
        ValueError: Raises a ValueError in case, it was not able to extract.

    Returns:
        OptunaSearchSpace: Returns the subspace for the given step.
    """
    prefix = delimiter.join([s.name for s in splits])

    prefix = f"{prefix}{delimiter}{step.name}" if prefix != "" else step.name
    choices = (s for s in splits if isinstance(s, Choice))
    subspace: OptunaSearchSpace

    # In case this step is supposed to be conditioned on a choice.
    if any(choices):
        raise ValueError("We currently do not support conditionals with Optuna.")

    # In case this step is a choice.
    if isinstance(step, Choice):
        raise ValueError("We currently do not support conditionals with Optuna.")

    if isinstance(step, Searchable):
        searchable: Searchable = step
        if searchable.space is None:
            return {}

        subspace = searchable.search_space

        if not isinstance(subspace, Mapping):
            raise ValueError(f"Expected space to be a mapping, got {type(subspace)}")

        if searchable.config is not None:
            subspace = {**subspace, **searchable.config}

        subspace = _extract_search_space(subspace, prefix=prefix, delimiter=delimiter)

    else:
        raise ValueError(f"Unknown type: {type(step)} of step: {step.name}")

    return subspace
