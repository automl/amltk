"""A simple random search optimizer.

This optimizer will sample from the space provided and return the results
without doing anything with them.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from typing_extensions import ParamSpec

from amltk.optimization.optimizer import Optimizer, Trial
from amltk.pipeline.sampler import Sampler
from amltk.randomness import as_rng
from amltk.types import Space

if TYPE_CHECKING:
    from amltk.types import Config, Seed

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
            | Callable[[Space, int], Config]
            | None
        ) = None,
        seed: Seed | None = None,
        duplicates: bool = False,
        max_sample_attempts: int = 50,
    ):
        """Initialize the optimizer.

        Args:
            space: The space to sample from.
            sampler: The sampler to use to sample from the space.
                If not provided, the sampler will be automatically found.

                * If a `Sampler` is provided, it will be used to sample from the
                    space.
                * If a `Callable` is provided, it will be used to sample from the space.

                    ```python
                    def my_callable_sampler(space, seed: int) -> Config: ...
                    ```

                    !!! warning "Deterministic behaviour"

                        This should return the same set of configurations given the same
                        seed for fully defined behaviour.


            seed: The seed to use for the sampler.
            duplicates: Whether to allow duplicate configurations.
            max_sample_attempts: The maximum number of attempts to sample a
                unique configuration. If this number is exceeded, an
                `ExhaustedError` will be raised. This parameter has no
                effect when `duplicates=True`.
        """
        self.space = space
        self.trial_count = 0
        self.seed = as_rng(seed) if seed is not None else None
        self.max_sample_attempts = max_sample_attempts

        # We store any configs we've seen to prevent duplicates
        self._configs_seen: list[Config] | None = [] if not duplicates else None

        if sampler is None:
            sampler = Sampler.find(space)
            if sampler is None:
                extra = "You can also provide a custom function to `sample=`."
                raise Sampler.NoSamplerFoundError(space, extra=extra)
            self.sampler = sampler

        elif isinstance(sampler, type) and issubclass(sampler, Sampler):
            self.sampler = sampler()
        elif isinstance(sampler, Sampler):
            self.sampler = sampler
        elif callable(sampler):
            self.sampler = FunctionalSampler(sampler)
        else:
            raise ValueError(
                f"Expected `sampler` to be a `Sampler` or `Callable`, got {sampler=}.",
            )

    def ask(self) -> Trial[RSTrialInfo]:
        """Sample from the space.

        Raises:
            ExhaustedError: If the sampler is exhausted of unique configs.
                Only possible to raise if `duplicates=False` (default).
        """
        name = f"random-{self.trial_count}"

        try:
            config = self.sampler.sample(
                self.space,
                seed=self.seed,
                duplicates=self._configs_seen,  # type: ignore
                max_attempts=self.max_sample_attempts,
            )
        except Sampler.GenerateUniqueConfigError as e:
            raise self.ExhaustedError(space=self.space) from e

        if self._configs_seen is not None:
            self._configs_seen.append(config)

        info = RSTrialInfo(name, self.trial_count, config)
        trial = Trial(
            name=name,
            config=config,
            info=info,
            seed=self.seed.integers(MAX_INT) if self.seed is not None else None,
        )
        self.trial_count = self.trial_count + 1
        return trial

    def tell(self, _: Trial.Report[RSTrialInfo]) -> None:
        """Do nothing with the report.

        ???+ note
            We do nothing with the report as it's random search
            and does not use the report to do anything useful.
        """

    class ExhaustedError(RuntimeError):
        """Raised when the sampler is exhausted of unique configs."""

        def __init__(self, space: Any):
            """Initialize the error."""
            self.space = space

        def __str__(self) -> str:
            return (
                f"Exhausted all unique configs in the space {self.space}."
                " Consider bumping up `max_sample_attempts=` or handling this"
                " error case."
            )


@dataclass
class FunctionalSampler(Sampler[Space]):
    """A wrapper for a functional sampler for use in
    [`RandomSearch`][amltk.optimization.RandomSearch].

    Attributes:
        f: The functional sampler to use.
    """

    f: Callable[[Space, int], Config]

    @classmethod
    def supports_sampling(cls, space: Any) -> bool:  # noqa: ARG003
        """Defaults to True for all spaces."""
        return True

    def copy(self, space: Space) -> Space:
        """Attempts it's best with a deepcopy."""
        return deepcopy(space)

    def _sample(
        self,
        space: Space,
        n: int = 1,
        seed: Seed | None = None,
    ) -> list[Config]:
        rng = as_rng(seed)
        return [self.f(space, rng.integers(MAX_INT)) for _ in range(n)]
