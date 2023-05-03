"""A module to interact with ConfigSpace."""
from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Constant
from ConfigSpace.hyperparameters import Hyperparameter

from byop.pipeline.space import SpaceAdapter
from byop.randomness import as_int

if TYPE_CHECKING:
    from byop.types import Seed


class ConfigSpaceAdapter(SpaceAdapter[ConfigurationSpace]):
    """An adapter following the [`SpaceAdapter`][byop.pipeline.SpaceAdapter] interface
    for interacting with ConfigSpace spaces.

    This includes parsing ConfigSpace spaces following the
    [`Parser`][byop.pipeline.Parser] interface and sampling from them with
    the [`Sampler`][byop.pipeline.Sampler] interface.
    """

    def parse_space(
        self,
        space: Any,
        config: Mapping[str, Any] | None = None,  # noqa: ARG002
    ) -> ConfigurationSpace:
        """See [`Parser.parse_space`][byop.pipeline.Parser.parse_space].

        ```python exec="true" source="material-block" result="python" title="A simple space"
        from byop.configspace import ConfigSpaceAdapter

        search_space = {
            "a": (1, 10),
            "b": (0.5, 9.0),
            "c": ["apple", "banana", "carrot"],
        }

        adapter = ConfigSpaceAdapter()
        space = adapter.parse(search_space)
        print(space)
        ```
        """  # noqa: E501
        if space is None:
            space = ConfigurationSpace()
        elif isinstance(space, dict):
            space = ConfigurationSpace(space)
        elif isinstance(space, Hyperparameter):
            space = ConfigurationSpace({space.name: space})
        elif isinstance(space, ConfigurationSpace):
            space = self.copy(space)
        else:
            TypeError(f"{space} is not parsable as a space")

        return space

    def set_seed(self, space: ConfigurationSpace, seed: Seed) -> ConfigurationSpace:
        """Set the seed for the space.

        ```python exec="true" source="material-block" result="python" title="Setting the seed"
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()

        space = adapter.parse({ "a": (1, 10) })
        adapter.set_seed(space, seed=42)

        seeded_value_for_a = adapter.sample(space)
        print(seeded_value_for_a)
        ```

        Args:
            space: The space to set the seed for.
            seed: The seed to set.
        """  # noqa: E501
        space.seed(seed)
        return space

    def insert(
        self,
        space: ConfigurationSpace,
        subspace: ConfigurationSpace,
        *,
        prefix_delim: tuple[str, str] | None = None,
    ) -> ConfigurationSpace:
        """See [`Parser.insert`][byop.pipeline.Parser.insert].

        ```python exec="true" source="material-block" result="python" title="Inserting one space into another"
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()

        space_1 = adapter.parse({ "a": (1, 10) })
        space_2 = adapter.parse({ "b": (10.5, 100.5) })
        space_3 = adapter.parse({ "c": ["apple", "banana", "carrot"] })

        space = adapter.empty()
        adapter.insert(space, space_1)
        adapter.insert(space, space_2)
        adapter.insert(space, space_3, prefix_delim=("fruit", ":"))

        print(space)
        ```
        """  # noqa: E501
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
        """See [`Parser.empty`][byop.pipeline.Parser.empty].

        ```python exec="true" source="material-block" result="python" title="Getting an empty space"
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()
        empty_space = adapter.empty()
        print(empty_space)
        ```
        """  # noqa: E501
        return ConfigurationSpace()

    def condition(
        self,
        choice_name: str,
        delim: str,
        spaces: dict[str, ConfigurationSpace],
        weights: Sequence[float] | None = None,
    ) -> ConfigurationSpace:
        """See [`Parser.condition`][byop.pipeline.Parser.condition].

        ```python exec="true" source="material-block" result="python" title="Conditioning a space"
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()

        space_a = adapter.parse({ "a": (1, 10) })
        space_b = adapter.parse({ "b": (200, 300) })

        space = adapter.condition(
            choice_name="letter",
            delim=":",
            spaces={ "a": space_a, "b": space_b }
        )
        print(space)
        ```
        """  # noqa: E501
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
        """See [`Sampler._sample`][byop.pipeline.Sampler._sample]."""
        if seed:
            seed_int = as_int(seed)
            self.set_seed(space, seed_int)

        if n == 1:
            return [dict(space.sample_configuration())]

        return [dict(c) for c in space.sample_configuration(n)]

    def copy(self, space: ConfigurationSpace) -> ConfigurationSpace:
        """See [`Sampler.copy`][byop.pipeline.Sampler.copy].

        ```python exec="true" source="material-block" result="python" title="Copying a space"
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()

        space_original = adapter.parse({ "a": (1, 10) })
        space_copy = adapter.copy(space_original)

        print(space_copy)
        ```
        """  # noqa: E501
        return deepcopy(space)

    @classmethod
    def supports_sampling(cls, space: Any) -> bool:
        """See [`Sampler.supports_sampling`][byop.pipeline.Sampler.supports_sampling].

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
