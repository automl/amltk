"""A module to help construct a configuration space for a pipeline.

``python
from byop.pipeline import Pipeline
from byop.spaces.configspace import generate_configspace

pipeline = Pipeline(...)
configspace = generate_configspace(pipeline)
```
"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Constant
from ConfigSpace.hyperparameters import Hyperparameter

from byop.pipeline.space import SpaceAdapter
from byop.randomness import as_int

if TYPE_CHECKING:
    from byop.types import Seed


class ConfigSpaceAdapter(SpaceAdapter[ConfigurationSpace]):
    """A sampler for a search space."""

    def parse_space(
        self,
        space: Any,
        config: Mapping[str, Any] | None = None,  # noqa: ARG002
    ) -> ConfigurationSpace:
        """See [`Parser.parse_space`][byop.pipeline.Parser.parse_space]."""
        if space is None:
            space = ConfigurationSpace()
        elif isinstance(space, dict):
            space = ConfigurationSpace(space)
        elif isinstance(space, Hyperparameter):
            space = ConfigurationSpace({space.name: space})
        elif isinstance(space, ConfigurationSpace):
            space = space
        else:
            TypeError(f"{space} is not parsable as a space")

        return space

    def set_seed(self, space: ConfigurationSpace, seed: Seed) -> ConfigurationSpace:
        """Set the seed for the space.

        Args:
            space: The space to set the seed for.
            seed: The seed to set.
        """
        space.seed(seed)
        return space

    def insert(
        self,
        space: ConfigurationSpace,
        subspace: ConfigurationSpace,
        *,
        prefix_delim: tuple[str, str] | None = None,
    ) -> ConfigurationSpace:
        """See [`Parser.insert`][byop.pipeline.Parser.insert]."""
        if prefix_delim is None:
            prefix_delim = ("", "")

        prefix, delim = prefix_delim

        space.add_configuration_space(
            prefix=prefix,
            configuration_space=subspace,
            delimiter=delim,
        )
        return space

    def empty(self) -> ConfigurationSpace:
        """See [`Parser.empty`][byop.pipeline.Parser.empty]."""
        return ConfigurationSpace()

    def condition(
        self,
        choice_name: str,
        delim: str,
        spaces: dict[str, ConfigurationSpace],
        weights: Sequence[float] | None = None,
    ) -> ConfigurationSpace:
        """See [`Parser.condition`][byop.pipeline.Parser.condition]."""
        space = ConfigurationSpace()

        items = list(spaces.keys())
        choice = Categorical(choice_name, items=items, weights=weights)
        space.add_hyperparameter(choice)

        for key, subspace in spaces.items():
            space.add_configuration_space(
                prefix=choice_name,
                configuration_space=subspace,
                parent_hyperparameter={"parent": choice, "value": key},
                delimiter=delim,
            )
        return space

    def _sample(
        self,
        space: ConfigurationSpace,
        n: int = 1,
        seed: Seed | None = None,
    ) -> list[Configuration]:
        """Sample a configuration from the given space."""
        if seed:
            seed_int = as_int(seed)
            self.set_seed(space, seed_int)

        if n == 1:
            return [space.sample_configuration()]

        return cast(list, space.sample_configuration(n))

    def copy(self, space: ConfigurationSpace) -> ConfigurationSpace:
        """Copy the space."""
        return deepcopy(space)

    @classmethod
    def supports_sampling(cls, space: Any) -> bool:
        """Check if the space is a ConfigurationSpace.

        Args:
            space: The space to check.

        Returns:
            True if the space is a ConfigurationSpace.
        """
        return isinstance(space, ConfigurationSpace)

    @classmethod
    def remove_hyperparameter(
        cls,
        name: str,
        space: ConfigurationSpace,
    ) -> ConfigurationSpace:
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

    @classmethod
    def replace_constants(
        cls,
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
                space = cls.remove_hyperparameter(key, space)

            if not isinstance(value, bool):
                hp = Constant(key, value)
                space.add_hyperparameter(hp)

        return space
