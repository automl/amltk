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

HyperparameterType: TypeAlias = int | str | float
OptunaConfig: TypeAlias = dict[str, HyperparameterType]
OptunaSearchSpace: TypeAlias = dict[str, BaseDistribution]
N_RANGE = 2


def _convert_hp_to_optuna_distribution(
    hp: tuple | list | HyperparameterType, name: str
) -> BaseDistribution:
    if isinstance(hp, tuple):
        if len(hp) != N_RANGE:
            raise ParseError(f"{name} must be (lower, upper) bound, got {hp}")
        lower, upper = hp
        if type(lower) != type(upper):
            raise ParseError(
                f"Expected {name} to have same type for lower and upper bound,"
                f"got lower: {type(lower)}, upper: {type(upper)}."
            )

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
    elif isinstance(hp, HyperparameterType):  # type: ignore[misc, arg-type]
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
    search_space: OptunaSearchSpace = {}
    for splits, _, step in pipeline.walk():
        _splits = splits if splits is not None else []
        subspace = _process_step(_splits, step)
        if subspace is not None:
            search_space.update(subspace)
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
        ParseError: Raises a ParseError in case, it was not able to extract.

    Returns:
        OptunaSearchSpace: Returns the subspace for the given step.
    """
    prefix = delimiter.join([s.name for s in splits])

    prefix = f"{prefix}{delimiter}{step.name}" if prefix != "" else step.name
    choices = (s for s in splits if isinstance(s, Choice))

    # In case this step is supposed to be conditioned on a choice.
    if any(choices):
        raise ParseError("We currently do not support conditionals with Optuna.")

    # In case this step is a choice.
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
        raise ParseError(f"Unknown type: {type(step)} of step: {step.name}")

    return subspace
