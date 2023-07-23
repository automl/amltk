"""The base definition of a Sampler.

It's primary role is to allow sampling from a particular Space.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, cast, overload

from more_itertools import first, first_true, seekable

from amltk.exceptions import safe_map
from amltk.randomness import as_int, as_rng
from amltk.types import Config, Seed, Space

logger = logging.getLogger(__name__)


class Sampler(ABC, Generic[Space]):
    """A sampler to sample configs from a search space.

    This class is a sampler for a given Space type, providing functionality
    to sample from the space. To implement a new sampler, subclass this class
    and implement the following methods:

    !!! example "Abstract Methods"

        * [`supports_sampling`][amltk.pipeline.sampler.Sampler.supports_sampling]:
            Check if the sampler supports sampling from a given Space.
        * [`_sample`][amltk.pipeline.Sampler._sample]: Sample from the
            given Space, given a specific seed and number of samples. Should
            ideally be deterministic given a pair `(seed, n)`.
            This is used in the [`sample`][amltk.pipeline.sampler.Sampler.sample]
            method.
        * [`copy`][amltk.pipeline.sampler.Sampler.copy]: Copy a Space to get
            and identical space.

        Please see the documentation for these methods for more information.


    See Also:
        * [`SpaceAdapter`][amltk.pipeline.space.SpaceAdapter]
            Together with implementing the [`Parser`][amltk.pipeline.Parser]
            interface, this class provides a complete adapter for a given search space.
    """

    @classmethod
    def default_samplers(cls, space: Any) -> list[Sampler]:
        """Get the default samplers compatible with a given space.

        Args:
            space: The space to sample from.

        Returns:
            A list of samplers that can sample from the given space.
        """
        samplers: list[Sampler] = []
        adapter: Sampler
        try:
            from amltk.configspace import ConfigSpaceAdapter

            adapter = ConfigSpaceAdapter()
            if adapter.supports_sampling(space):
                samplers.append(adapter)

            samplers.append(adapter)
        except ImportError:
            logger.debug("ConfigSpace not installed for sampling, skipping")

        try:
            from amltk.optuna import OptunaSpaceAdapter

            adapter = OptunaSpaceAdapter()
            if adapter.supports_sampling(space):
                samplers.append(adapter)

            samplers.append(adapter)
        except ImportError:
            logger.debug("Optuna not installed for sampling, skipping")

        return samplers

    @classmethod
    def find(cls, space: Any) -> Sampler | None:
        """Find a sampler that supports the given space.

        Args:
            space: The space to sample from.

        Returns:
            The first sampler that supports the given space, or None if no
            sampler supports the given space.
        """
        return first(
            (
                sampler
                for sampler in Sampler.default_samplers(space)
                if sampler.supports_sampling(space)
            ),
            default=None,
        )

    @overload
    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = ...,
        *,
        n: None = None,
        seed: Seed | None = ...,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> Config:
        ...

    @overload
    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = ...,
        *,
        n: int,
        seed: Seed | None = ...,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> list[Config]:
        ...

    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: type[Sampler[Space]] | Sampler[Space] | None = None,
        *,
        n: int | None = None,
        seed: Seed | None = None,
        duplicates: bool | Iterable[Config] = False,
        max_attempts: int | None = 10,
    ) -> Config | list[Config]:
        """Attempt to sample a pipeline with the default samplers.

        Args:
            space: The space to sample from.
            sampler: The sampler to use. If None, the default samplers will be
                used.
            n: The number of samples to return. If None, a single sample will
                be returned.
            seed: The seed to use for sampling.
            duplicates: If True, allow duplicate samples. If False, make
                sure all samples are unique. If a Iterable, make sure all
                samples are unique and not in the Iterable.
            max_attempts: The number of times to attempt sampling unique
                configurations before giving up. If `None` will keep
                sampling forever until satisfied.

        Returns:
            A single sample if `n` is None, otherwise a list of samples.
        """
        if sampler is None:
            samplers = cls.default_samplers(space)
        elif isinstance(sampler, Sampler):
            samplers = [sampler]
        else:
            samplers = [sampler()]

        if not any(samplers):
            raise RuntimeError(
                "Found no possible sampler to use. Have you tried installing any of:"
                "\n* ConfigSpace"
                "\n* Optuna"
                "\nPlease see the integration documentation for more info, especially"
                "\nif using an optimizer which often requires a specific search space."
                "\nUsually just installing the optimizer will work.",
            )

        def _sample(_sampler: Sampler[Space]) -> Config | list[Config]:
            _space = _sampler.copy(space)
            return _sampler.sample(
                space=_space,
                n=n,
                seed=seed,
                duplicates=duplicates,
                max_attempts=max_attempts,
            )

        # Wrap in seekable so we don't evaluate all of them, only as
        # far as we need to get a succesful parse.
        results_itr = seekable(safe_map(_sample, samplers))

        is_result = lambda r: not (isinstance(r, tuple) and isinstance(r[0], Exception))

        # Progress the iterator until we get a successful sample
        samples = first_true(results_itr, default=False, pred=is_result)

        # If we didn't get a succesful parse, raise the appropriate error
        if samples is False:
            results_itr.seek(0)  # Reset to start of iterator
            errors = cast(list[Exception], list(results_itr))
            raise Sampler.FailedSamplingError(sampler=samplers, error=errors)

        assert not isinstance(samples, (tuple, bool))
        return samples

    @overload
    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: None = None,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> Config:
        ...

    @overload
    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: int,
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> list[Config]:
        ...

    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: int | None = None,
        duplicates: bool | Iterable[Config] = False,
        max_attempts: int | None = 10,
    ) -> Config | list[Config]:
        """Sample a configuration from the given space.

        Args:
            space: The space to sample from.
            seed: The seed to use for sampling.
            n: The number of configurations to sample.
            duplicates: If True, allow duplicate samples. If False, make
                sure all samples are unique. If a Iterable, make sure all
                samples are unique and not in the Iterable.
            max_attempts: The number of times to attempt sampling unique
                configurations before giving up. If `None` will keep
                sampling forever until satisfied.

        Returns:
            A single sample if `n` is None, otherwise a list of samples.
            If `duplicates` is not True and we fail to sample.
        """
        _n = 1 if n is None else n
        rng = as_rng(seed)

        if duplicates is True:
            samples = self._sample(space=space, n=_n, seed=rng)
            return samples[0] if n is None else samples

        # NOTE: We use a list here as Config's could be a dict
        #   which are not hashable. We rely on equality checks
        seen = list(duplicates) if isinstance(duplicates, Iterable) else []

        samples = []
        _max_attempts: int = max_attempts if max_attempts is not None else 2**32
        rng = as_rng(seed)
        for _ in range(_max_attempts):
            next_seed = as_int(rng)
            _samples = self._sample(space=space, n=_n, seed=next_seed)

            for s in _samples:
                if s in seen:
                    continue
                samples.append(s)
                seen.append(s)

            if len(samples) >= _n:
                break

        if len(samples) != _n:
            raise Sampler.GenerateUniqueConfigError(
                n=_n,
                max_attempts=_max_attempts,
                seen=seen,
            )

        return samples[0] if n is None else samples[:n]

    @classmethod
    @abstractmethod
    def supports_sampling(cls, space: Any) -> bool:
        """Check if the space is supported for sampling.

        Args:
            space: The space to check.

        Returns:
            True if the space is supported, False otherwise.
        """
        ...

    @abstractmethod
    def copy(self, space: Space) -> Space:
        """Copy the space.

        Args:
            space: The space to copy.

        Returns:
            A copy of the space.
        """
        ...

    @abstractmethod
    def _sample(
        self,
        space: Space,
        n: int = 1,
        seed: Seed | None = None,
    ) -> list[Config]:
        """Sample a configuration from the given space.

        Args:
            space: The space to sample from.
            n: The number of configurations to sample.
            seed: The seed to use for sampling.

        Returns:
            A list of samples.
        """
        ...

    class NoSamplerFoundError(ValueError):
        """Error when no sampler is found for a given space."""

        def __init__(self, space: Any, extra: str | None = None):
            """Create a new no sampler found error.

            Args:
                space: The space that no sampler was found for
                extra: Any extra information to add to the error message.
            """
            self.space = space
            self.extra = extra

        def __str__(self) -> str:
            msg = (
                f"No sampler found for space of type={type(self.space)}."
                " Do you have the correct integrations installed? If none"
                " exist for your space, you can create your own sampler."
            )
            if self.extra:
                msg += f" {self.extra}"

            return msg

    class GenerateUniqueConfigError(RuntimeError):
        """Error when a Sampler fails to sample a unique configuration."""

        def __init__(self, n: int, max_attempts: int, seen: list[Config]):
            """Create a new sample unique config error.

            Args:
                n: The number of unique configs that were to sample.
                max_attempts: The maximum number of attempts made to sample
                    `n` unique configurations.
                seen: The configs seen during sampling.
            """
            self.n = n
            self.max_attempts = max_attempts
            self.seen = seen

        def __str__(self) -> str:
            n = self.n
            max_attempts = self.max_attempts
            seen = self.seen
            return (
                f"Could not find {n=} unique configs after {max_attempts=} attempts."
                "\nYou could try increasing the `max_attempts=` parameter."
                f"\n {len(seen)} seen: {seen=}"
            )

    class FailedSamplingError(RuntimeError):
        """Error for when a Sampler fails to sample from a Space."""

        @overload
        def __init__(self, sampler: Sampler, error: Exception):
            ...

        @overload
        def __init__(self, sampler: list[Sampler], error: list[Exception]):
            ...

        def __init__(
            self,
            sampler: Sampler | list[Sampler],
            error: Exception | list[Exception],
        ):
            """Create a new sampler error.

            Args:
                sampler: The sampler(s) that failed.
                error: The error(s) that was raised.

            Raises:
                ValueError: If sampler is a list, exception must be a list of the
                    same length.
            """
            if isinstance(sampler, list) and (
                not (isinstance(error, list) and len(sampler) == len(error))
            ):
                raise ValueError(
                    "If sampler is a list, `error` must be a list of the same length."
                    f"Got {sampler=} and {error=} .",
                )

            self.sampler = sampler
            self.error = error

        def __str__(self) -> str:
            if isinstance(self.sampler, list):
                msg = "\n\n".join(
                    f"Failed to sample with {p}:"
                    + "\n "
                    + f"{e.__class__.__name__}: {e}"
                    for p, e in zip(self.sampler, self.error)  # type: ignore
                )
            else:
                msg = (
                    f"Failed to sample with {self.sampler}:"
                    + "\n"
                    + f"{self.error.__class__.__name__}: {self.error}"
                )

            return msg
