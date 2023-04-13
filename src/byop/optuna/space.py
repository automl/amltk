"""A module for utilities for spaces defined by Optuna.

The notable class is [`OptunaSpaceAdapter`][byop.optuna.OptunaSpaceAdapter].


```python hl_lines="8 9 10 13 15 16"
from byop.pipeline import step
from byop.optuna import OptunaSpaceAdapter

item = step(
    "name",
    ...,
    space={
        "myint": (1, 10),  # (1)!
        "myfloat": (1.0, 10.0)  # (2)!
        "mycategorical": ["a", "b", "c"],  # (3)!
    }
)
adapter = OptunaSpaceAdapter()  # (6)!

optuna_space = item.space(parser=adapter)  # (4)!
config = item.sample(sampler=adapter)  # (5)!

configured_item = item.configure(config)
configured_item.build()
```

1. `myint` will be an integer between 1 and 10.
2. `myfloat` will be a float between 1.0 and 10.0.
3. `mycategorical` will be a categorical variable with values `a`, `b`, and `c`.
4. Pass the `adapter` to [`space()`][byop.pipeline.Step.space] to get the optuna space.
    It will be a dictionary mapping the name of the hyperparameter to the optuna
    distribution.
5. Pass the `adapter` to [`sample()`][byop.pipeline.Step.sample] to get a sample config.
    It will be a dictionary mapping the name of the hyperparameter to the sampled value.
6. Create an instance of [`OptunaSpaceAdapter`][byop.optuna.OptunaSpaceAdapter] which
    will be used to parse the space and sample from it.

    !!! note "Note"

        The `OptunaSpaceAdapter` is a [`SpaceAdapter`][byop.pipeline.SpaceAdapter] which
        means that it can be used to parse any space and sample from it. It implements
        the [`Parser`][byop.pipeline.Parser] and [`Sampler`][byop.pipeline.Sampler]
        interfaces.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from byop.configspace.space import as_int
from byop.pipeline.space import SpaceAdapter
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import RandomSampler

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from byop.types import Config, Seed

OptunaSearchSpace: TypeAlias = Dict[str, BaseDistribution]


class OptunaSpaceAdapter(SpaceAdapter[OptunaSearchSpace]):
    """An Optuna adapter to allow for parsing Optuna spaces and sampling from them."""

    def parse_space(
        self,
        space: Any,
        config: Mapping[str, Any] | None = None,
    ) -> OptunaSearchSpace:
        """See [`Parser.parse_space`][byop.pipeline.Parser.parse_space]."""
        if not isinstance(space, Mapping):
            raise ValueError("Can only parse mappings with Optuna but got {space=}")

        parsed_space = {
            name: self._convert_hp_to_optuna_distribution(name=name, hp=hp)
            for name, hp in space.items()
        }
        for name, value in (config or {}).items():
            parsed_space[name] = CategoricalDistribution([value])

        return parsed_space

    def insert(
        self,
        space: OptunaSearchSpace,
        subspace: OptunaSearchSpace,
        *,
        prefix_delim: tuple[str, str] | None = None,
    ) -> OptunaSearchSpace:
        """See [`Parser.insert`][byop.pipeline.Parser.insert]."""
        if prefix_delim is None:
            prefix_delim = ("", "")

        prefix, delim = prefix_delim

        space.update({f"{prefix}{delim}{name}": hp for name, hp in subspace.items()})

        return space

    def condition(
        self,
        choice_name: str,
        delim: str,
        spaces: dict[str, OptunaSearchSpace],
        weights: Sequence[float] | None = None,
    ) -> OptunaSearchSpace:
        """See [`Parser.condition`][byop.pipeline.Parser.condition]."""
        # TODO(eddiebergman): Might be possible to implement this but it requires some
        # toying around with options to various Samplers in the Optimizer used.
        raise NotImplementedError(
            f"Conditions (from {choice_name}) not supported with Optuna",
        )

    def empty(self) -> OptunaSearchSpace:
        """Return an empty space."""
        return {}

    def copy(self, space: OptunaSearchSpace) -> OptunaSearchSpace:
        """Copy the space."""
        return deepcopy(space)

    def _sample(
        self,
        space: OptunaSearchSpace,
        n: int = 1,
        seed: Seed | None = None,
    ) -> list[Config]:
        """Sample n configs from the space.

        Args:
            space: The space to sample from.
            n: The number of configs to sample.
            seed: The seed to use for sampling.

        Returns:
            A list of configs sampled from the space.
        """
        seed_int = as_int(seed)
        sampler = RandomSampler(seed=seed_int)

        # Can be used because `sample_independant` doesn't use the study or trial
        study: Any = None
        trial: Any = None

        # Sample n configs
        configs: list[Config] = [
            {
                name: sampler.sample_independent(study, trial, name, dist)
                for name, dist in space.items()
            }
            for _ in range(n)
        ]
        return configs

    @classmethod
    def _convert_hp_to_optuna_distribution(
        cls,
        name: str,
        hp: tuple | Sequence | int | str | float | BaseDistribution,
    ) -> BaseDistribution:
        if isinstance(hp, BaseDistribution):
            return hp

        # If it's an allowed type, it's a constant
        # TODO: Not sure if this makes sense to be honest
        if isinstance(hp, (int, str, float)):
            return CategoricalDistribution([hp])

        if isinstance(hp, tuple) and len(hp) == 2:  # noqa: PLR2004
            lower, upper = hp
            if type(lower) != type(upper):
                raise ValueError(
                    f"Expected {name} to have same type for lower and upper bound,"
                    f"got lower: {type(lower)}, upper: {type(upper)}.",
                )

            if isinstance(lower, float):
                return FloatDistribution(lower, upper)

            return IntDistribution(lower, upper)

        # Sequences
        if isinstance(hp, Sequence):
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            return CategoricalDistribution(hp)

        raise ValueError(
            f"Expected hyperparameter value for {name} to be one of "
            "tuple | list | int | str | float | Optuna.BaseDistribution,"
            f" got {type(hp)}",
        )

    @classmethod
    def supports_sampling(cls, space: Any) -> bool:
        """Supports sampling from a mapping where every value is a
        [`BaseDistribution`][optuna.distributions].

        Args:
            space: The space to check.

        Returns:
            Whether the space is supported.
        """
        return isinstance(space, Mapping) and all(
            isinstance(hp, BaseDistribution) for hp in space.values()
        )
