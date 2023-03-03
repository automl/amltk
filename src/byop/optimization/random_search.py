"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, ParamSpec, TypeVar

from byop.optimization.optimizer import Optimizer, Trial, TrialReport
from byop.samplers import Sampler
from byop.types import Config, Seed, Space

P = ParamSpec("P")
Q = ParamSpec("Q")
Result = TypeVar("Result")


@dataclass
class RSTrialInfo(Generic[Config]):
    """The information about a random search trial.

    Args:
        name: The name of the trial.
        trial_number: The number of the trial.
        config: The configuration sampled from the space.
    """

    name: str
    trial_number: int
    config: Config


class RandomSearch(Optimizer[RSTrialInfo[Config], Config]):
    """A random search optimizer."""

    def __init__(
        self,
        *,
        space: Space,
        sampler: Callable[[Space], Config] | Sampler[Space, Config] | None = None,
        seed: Seed | None = None,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            sampler: The sampler to use to sample from the space.
                If not provided, the sampler will be automatically found.
            seed: The seed to use for the sampler, if no sampler is provided.
        """
        self.space = space
        self.trial_count = 0

        self.sampler: Callable[[], Config]
        if sampler is None:
            sampler_cls: type[Sampler[Space, Config]] = Sampler.find(space)
            self.sampler = sampler_cls(space, seed=seed)
        elif isinstance(sampler, Sampler):
            self.sampler = sampler
        else:
            self.sampler = partial(sampler, space)

    def ask(self) -> Trial[RSTrialInfo[Config], Config]:
        """Sample from the space."""
        config = self.sampler()
        name = f"random-{self.trial_count}"
        info = RSTrialInfo(name, self.trial_count, config)
        trial = Trial(name=name, config=config, info=info)
        self.trial_count = self.trial_count + 1
        return trial

    def tell(self, _: TrialReport[RSTrialInfo[Config], Config]) -> None:
        """Do nothing with the report.

        ???+ note
            We do nothing with the report as it's random search
            and does not use the report to do anything useful.

        Args:
            report: The report of the trial.
        """
