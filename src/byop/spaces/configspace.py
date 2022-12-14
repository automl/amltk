"""A module to help construct a configuration space for a pipeline.

```python
from byop.pipeline import Pipeline
from byop.spaces.configspace import generate_configspace

pipeline = Pipeline(...)
configspace = generate_configspace(pipeline)
```
"""
from __future__ import annotations

from copy import copy
from itertools import takewhile
from typing import Any, Mapping

import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Constant, EqualsCondition
from more_itertools import first, last

from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split, Step


def remove_hyperparameter(name: str, space: ConfigurationSpace) -> ConfigurationSpace:
    """A new configuration space with the hyperparameter removed.

    Essentially copies hp over and fails if there is conditionals or forbiddens
    """
    if name not in space._hyperparameters:
        raise ValueError(f"{name} not in {space}")

    # Copying conditionals only work on objects and not named entities
    # Seeing as we copy objects and don't use the originals, transfering these
    # to the new objects is a bit tedious, possible but not required at this time
    # ... same goes for forbiddens
    assert name not in space._conditionals, "Can't handle conditionals"
    assert not any(
        name != f.hyperparameter.name for f in space.get_forbiddens()
    ), "Can't handle forbiddens"

    hps = [copy(hp) for hp in space.get_hyperparameters() if hp.name != name]

    if isinstance(space.random, np.random.RandomState):
        new_seed = space.random.randint(2**32 - 1)
    else:
        new_seed = copy(space.random)

    new_space = ConfigurationSpace(
        # TODO: not sure if this will have implications, assuming not
        seed=new_seed,
        name=copy(space.name),
        meta=copy(space.meta),
    )
    new_space.add_hyperparameters(hps)
    return new_space


def replace_constants(
    config: Mapping[str, Any],
    space: ConfigurationSpace,
) -> ConfigurationSpace:
    """Search the config for any hyperparameters that are in the space and need.

    to be replaced with a constant.

    Args:
        config: The configuration associated with a step, which may have
            overlaps with the ConfigurationSpace
        space: The space to remove overlapping parameters from

    Returns:
        ConfigurationSpace: A copy of the space with the hyperparameters replaced
    """
    for key, value in config.items():
        if key in space._hyperparameters:
            space = remove_hyperparameter(key, space)
            space.add_hyperparameter(Constant(key, value))
    return space


def generate_configspace(
    pipeline: Pipeline,
    seed: int | np.random.RandomState | np.random.BitGenerator | None = None,
) -> ConfigurationSpace:
    """The space for this given pipeline.

    Args:
        pipeline: The pipeline to generate the space for
        seed: The seed to use for the ConfigurationSpace

    Returns:
        ConfigurationSpace
    """
    cs = ConfigurationSpace(seed=seed)
    for splits, parents, step in pipeline.walk():
        parents = parents if parents is not None else []
        splits = splits if splits is not None else []
        _process_step(splits, parents, step, cs)
    return cs


def _process_step(
    splits: list[Split],
    parents: list[Step],
    step: Step,
    space: ConfigurationSpace,
    *,
    delimiter: str = ":",
) -> None:
    prefix = delimiter.join([s.name for s in splits])

    choices = (s for s in splits if isinstance(s, Choice))

    # If there is a choice leading up to this step,
    # we create a condition that this step is only active while the
    # last encounter choice is active
    last_choice = last(choices, None)
    condition: dict | None
    if last_choice:
        # Get all steps leading up to the last choice found
        pre_choice = takewhile(lambda step: step != last_choice, splits)
        last_choice_name = delimiter.join(
            [step.name for step in pre_choice] + [last_choice.name]
        )
        choice_hp = space[last_choice_name]  # ! This relies on traversal order to exist

        # Conditioned on it's first parent in the chain being active
        # by the choice
        first_parent = first(parents, step)
        condition = {"parent": choice_hp, "value": first_parent.name}
    else:
        condition = None

    if isinstance(step, (Component, Split)) and step.space is not None:
        subspace = step.space
        if step.config is not None:
            subspace = replace_constants(step.config, subspace)

        space.add_configuration_space(
            prefix=prefix,
            configuration_space=subspace,
            delimiter=delimiter if prefix != "" else "",
            parent_hyperparameter=condition,
        )

    elif isinstance(step, Choice):
        if step.space is not None:
            raise ValueError(
                f"Not currently supported to have a choice with a search space, {step=}"
            )
        name = delimiter.join([prefix, step.name]) if prefix != "" else step.name
        choice_hp = Categorical(
            name=name,
            items=[choice.name for choice in step.paths],
            weights=step.weights,
        )
        space.add_hyperparameter(choice_hp)
        if condition:
            cond = EqualsCondition(choice_hp, condition["parent"], condition["value"])
            space.add_condition(cond)

    else:
        pass
