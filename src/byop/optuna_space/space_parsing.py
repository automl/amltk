"""A module to help construct an optuna search space for a pipeline.

``python
from byop.pipeline import Pipeline
from byop.optuna_space import generate_optuna_search_space

pipeline = Pipeline(...)
configspace = generate_optuna_search_space(pipeline)
```
"""
from __future__ import annotations

from typing import Any, TypeAlias

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from byop.parsing.space_parsers.space_parser import ParseError
from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split, Step

HYPERPARAMETER_TYPE = int | str | float
OPTUNA_CONFIG: TypeAlias = dict[str, HYPERPARAMETER_TYPE]
OPTUNA_SEARCH_SPACE: TypeAlias = dict[str, BaseDistribution]
N_RANGE = 2


def _convert_hp_to_optuna_distribution(
    hp: tuple | list | HYPERPARAMETER_TYPE, name: str
) -> BaseDistribution:
    if isinstance(hp, tuple):
        if len(hp) != N_RANGE:
            raise ParseError(f"{name} must be (lower, upper) bound, got {hp}")
        lower, upper = hp
        if isinstance(lower, float):
            real_hp = FloatDistribution(lower, upper)
        else:
            real_hp = IntDistribution(lower, upper)

    # Lists are categoricals
    elif isinstance(hp, list):
        if len(hp) == 0:
            raise ParseError(f"Can't have empty list for categorical {name}")

        real_hp = CategoricalDistribution(hp)

    # If it's an allowed type, it's a constant
    elif isinstance(hp, HYPERPARAMETER_TYPE):  # type: ignore[misc, arg-type]
        real_hp = CategoricalDistribution([hp])
    else:
        raise ParseError(
            f"Expected hyperparameter value for {name} to be one of "
            f"tuple | list | int | str | float, got {type(hp)}"
        )
    return real_hp


def _extract_search_space(
    space: dict[str, Any],
    prefix: str,
    delimiter: str,
) -> OPTUNA_SEARCH_SPACE:
    """Extracts "Define-and-run" search space compatible with
    Optuna study for a given step.

    Args:
        space: search space of the step.
        prefix: prefix that is added to the name of each hyperparameter,
            usually the name of parent steps.
        delimiter: Symbol used to join the prefix with the name of the hyperparameter.

    Returns:
        OPTUNA_SEARCH_SPACE
    """
    search_space: OPTUNA_SEARCH_SPACE = {}
    for name, hp in space.items():
        if isinstance(hp, BaseDistribution):
            subspace = hp
        else:
            subspace = _convert_hp_to_optuna_distribution(hp=hp, name=name)
        search_space[f"{prefix}{delimiter}{name}"] = subspace
    return search_space


def generate_optuna_search_space(
    pipeline: Pipeline,
) -> OPTUNA_SEARCH_SPACE:
    """Generates the search space for the given pipeline.

    Args:
        pipeline: The pipeline to generate the space for.

    Returns:
        OPTUNA_SEARCH_SPACE
    """
    search_space: OPTUNA_SEARCH_SPACE = {}
    for splits, _, step in pipeline.walk():
        _splits = splits if splits is not None else []
        subspace = _process_step(_splits, step)
        if subspace is not None:
            search_space = {**search_space, **subspace}
    return search_space


def _process_step(
    splits: list[Split],
    step: Step,
    *,
    delimiter: str = ":",
) -> OPTUNA_SEARCH_SPACE:
    """Returns the subspace for the given step of the pipeline.

    Args:
        splits (list[Split]): The list of steps in the pipeline
            where the flow of the data is split to reach this step.
        step (Step): step to extract search space from
        delimiter (str): Delimiter used to separate different steps in
            the hyperparameter name. Defaults to ":".

    Returns:
        OPTUNA_SEARCH_SPACE | ParseError: Returns the subspace or raises a ParseError
            in case, it was not able to extract.
    """
    prefix = delimiter.join([s.name for s in splits]) if len(splits) > 0 else ""

    prefix = f"{prefix}{delimiter}{step.name}" if bool(prefix) else step.name
    choices = (s for s in splits if isinstance(s, Choice))

    # In case this step is supposed to be conditioned on a choice.
    if any(choices):
        raise ParseError("We currently do not support conditionals with Optuna.")

    if isinstance(step, Choice):
        raise ParseError("We currently do not support conditionals with Optuna.")

    if isinstance(step, (Component, Split)):
        if step.space is not None:
            subspace = step.space

            if step.config is not None:
                subspace = {**subspace, **step.config}
            subspace = _extract_search_space(
                subspace, prefix=prefix, delimiter=delimiter
            )
        else:
            subspace = {}

    else:
        raise ParseError(f"Unknown type: {type(step)} of step")

    return subspace
