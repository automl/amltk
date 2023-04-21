"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar
from typing_extensions import ParamSpec

from byop.optimization.optimizer import Optimizer, Trial
from byop.pipeline.sampler import Sampler
from byop.randomness import as_rng

if TYPE_CHECKING:
    from byop.types import Config, Seed, Space

P = ParamSpec("P")
Q = ParamSpec("Q")
Result = TypeVar("Result")

MAX_INT = 2**32


@dataclass
class RSTrialInfo:
    """The information about a random search trial.

    Args:
        name: The name of the trial.
        trial_number: The number of the trial.
        config: The configuration sampled from the space.
    """

    name: str
    trial_number: int
    config: Config


class RandomSearch(Optimizer[RSTrialInfo]):
    """A random search optimizer."""

    def __init__(
        self,
        *,
        space: Space,
        sampler: (
            Sampler[Space]
            | type[Sampler[Space]]
            | Callable[[Space], Config]
            | Callable[[Space, int], Config]
            | None
        ) = None,
        seed: Seed | None = None,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            sampler: The sampler to use to sample from the space.
                If not provided, the sampler will be automatically found.
                * If a `Sampler` is provided, it will be used to sample from the
                    space.
                * If a `Callable` is provided, it will be used to sample from the
                    space. If providing a `seed`, the `Callable` must accept a
                    keyword argument `seed: int` which will be given an integer
                    generated from the seed given in the `__init__`.
            seed: The seed to use for the sampler.
        """
        self.space = space
        self.trial_count = 0
        self.seed = as_rng(seed) if seed is not None else None

        if sampler is None:
            sampler = Sampler.find(space)
        elif isinstance(sampler, type) and issubclass(sampler, Sampler):
            sampler = sampler()

        if isinstance(sampler, Sampler):
            self.sample_f = sampler.sample
        elif callable(sampler):
            self.sample_f = sampler  # type: ignore
        else:
            raise ValueError(
                f"Expected `sampler` to be a `Sampler` or `Callable`, got {sampler=}.",
            )

    def ask(self) -> Trial[RSTrialInfo]:
        """Sample from the space."""
        name = f"random-{self.trial_count}"

        # NOTE(eddiebergman): We validate this is correct
        # in the init and with the docstring. Any errors
        # that occur from here are considered user error
        if self.seed is None:
            config = self.sample_f(self.space)  # type: ignore
        else:
            try:
                seed_int = self.seed.integers(MAX_INT)
                config = self.sample_f(self.space, seed=seed_int)  # type: ignore
            except TypeError as e:
                msg = (
                    f"Expected `sampler={self.sample_f}` to accept a `seed`"
                    f" keyword argument: {e}"
                )
                raise TypeError(msg) from e

        info = RSTrialInfo(name, self.trial_count, config)
        trial = Trial(name=name, config=config, info=info)
        self.trial_count = self.trial_count + 1
        return trial

    def tell(self, _: Trial.Report[RSTrialInfo]) -> None:
        """Do nothing with the report.

        ???+ note
            We do nothing with the report as it's random search
            and does not use the report to do anything useful.
        """
