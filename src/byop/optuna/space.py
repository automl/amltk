"""A module to help construct an optuna search space for a pipeline.

``python
from byop.pipeline import Pipeline
from byop.optuna_space import generate_optuna_search_space
 = Pipeline(...)
configspace = generate_optuna_search_space(pipeline)
```
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
        """See [`Parser.parse_space`][byop.parsing.Parser.parse_space]."""
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
        """See [`Parser.insert`][byop.parsing.Parser.insert]."""
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
        """See [`Parser.condition`][byop.parsing.Parser.condition]."""
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
        [`BaseDistribution`][optuna.distributions.BaseDistribution].

        Args:
            space: The space to check.

        Returns:
            Whether the space is supported.
        """
        return isinstance(space, Mapping) and all(
            isinstance(hp, BaseDistribution) for hp in space.values()
        )
