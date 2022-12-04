from __future__ import annotations

from copy import copy
from itertools import chain
from typing import Any

import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Constant

from byop.pipeline import Choice, Searchable, Split
from byop.pipeline.step import Step


def remove_hyperparameter(name: str, space: ConfigurationSpace) -> ConfigurationSpace:
    """A new configuration space with the hyperparameter removed

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


def replace_hp(name: str, value: Any, space: ConfigurationSpace) -> ConfigurationSpace:
    """Replace a hyperparameter with a set value in a ConfigurationSpace

    Args:
        name: Name of the hyperparameter
        value: Value to use
        space: The space in which to replace it

    Returns:
        ConfigurationSpace: Copy of the space with the hyperparameter removed
    """
    space = remove_hyperparameter(name, space)
    space.add_hyperparameter(Constant(name, value))
    return space


def generate_configspace(
    head: Step,
    seed: int | np.random.RandomState | np.random.BitGenerator | None = None,
) -> ConfigurationSpace:
    """The space for this given pipeline

    Args:
        head: The head of the pipeline
        seed:

    Returns:
        ConfigurationSpace
    """
    cs = ConfigurationSpace(seed=seed)
    add_subspace = cs.add_configuration_space

    for node in chain([head], head.following):

        if isinstance(node, Searchable):
            space = node.space
            if any(node.config):
                for key, value in node.config.items():
                    space = replace_hp(key, value, space)
            add_subspace(node.name, space)

        elif isinstance(node, Split):
            for step in node.paths:
                add_subspace("", generate_configspace(step, seed), delimiter="")

        elif isinstance(node, Choice):
            names = [choice.name for choice in node.choices]
            choice_hp = Categorical(node.name, names, weights=node.weights)
            cs.add_hyperparameter(choice_hp)

            for choice in node.choices:
                condition = {"parent": choice_hp, "value": choice.name}
                add_subspace(
                    node.name,
                    generate_configspace(choice, seed),
                    ":",
                    condition,
                )

    return cs
