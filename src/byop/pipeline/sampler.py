"""The base definition of a Sampler.

It's primary role is to allow sampling from a particular Space.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, cast, overload

from more_itertools import first, first_true, seekable

from byop.exceptions import safe_map
from byop.types import Config, Seed, Space

logger = logging.getLogger(__name__)


class SamplerError(Exception):
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
                f"Failed to sample with {p}:" + "\n " + f"{e.__class__.__name__}: {e}"
                for p, e in zip(self.sampler, self.error)  # type: ignore
            )
        else:
            msg = (
                f"Failed to parse with {self.sampler}:"
                + "\n"
                + f"{self.error.__class__.__name__}: {self.error}"
            )

        return msg


class Sampler(ABC, Generic[Space]):
    """A sampler to sample configs from a search space.

    This class is a sampler for a given Space type, providing functionality
    to sample from the space. To implement a new sampler, subclass this class
    and implement the following methods:

    !!! example "Abstract Methods"

        * [`supports_sampling`][byop.pipeline.sampler.Sampler.supports_sampling]:
            Check if the sampler supports sampling from a given Space.
        * [`_sample`][byop.pipeline.sampler.Sampler._sample]: Sample from the
            given Space, given a specific seed and number of samples. Should
            ideally be deterministic given a pair `(seed, n)`.
            This is used in the [`sample`][byop.pipeline.sampler.Sampler.sample]
            method.
        * [`copy`][byop.pipeline.sampler.Sampler.copy]: Copy a Space to get
            and identical space.

        Please see the documentation for these methods for more information.


    See Also:
        * [`SpaceAdapter`][byop.pipeline.space.SpaceAdapter]
            Together with implementing the [`Sampler`][byop.pipeline.sampler.Sampler]
            interface, this class provides a complete adapter for a given search space.
    """

    SamplerError: ClassVar[type[SamplerError]] = SamplerError
    """The error to raise when a sampler fails to sample from a Space."""

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
            from byop.configspace import ConfigSpaceAdapter

            adapter = ConfigSpaceAdapter()
            if adapter.supports_sampling(space):
                samplers.append(adapter)

            samplers.append(adapter)
        except ImportError:
            logger.debug("ConfigSpace not installed for sampling, skipping")

        try:
            from byop.optuna import OptunaSpaceAdapter

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
        return first(Sampler.default_samplers(space), default=None)

    @overload
    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: Sampler[Space] | None = ...,
        *,
        n: None = None,
        seed: Seed | None = ...,
    ) -> Config:
        ...

    @overload
    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: Sampler[Space] | None = ...,
        *,
        n: int,
        seed: Seed | None = ...,
    ) -> list[Config]:
        ...

    @classmethod
    def try_sample(
        cls,
        space: Space,
        sampler: Sampler[Space] | None = None,
        *,
        n: int | None = None,
        seed: Seed | None = None,
    ) -> Config | list[Config]:
        """Attempt to sample a pipeline with the default samplers.

        Args:
            space: The space to sample from.
            sampler: The sampler to use. If None, the default samplers will be
                used.
            n: The number of samples to return. If None, a single sample will
                be returned.
            seed: The seed to use for sampling.

        Returns:
            A single sample if `n` is None, otherwise a list of samples.
        """
        samplers = [sampler] if sampler is not None else Sampler.default_samplers(space)

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
            return _sampler.sample(space=_space, n=n, seed=seed)

        # Wrap in seekable so we don't evaluate all of them, only as
        # far as we need to get a succesful parse.
        results_itr = seekable(safe_map(_sample, samplers, attached_tb=True))

        # Progress the iterator until we get a successful sample
        samples = first_true(
            results_itr,
            default=False,
            pred=lambda result: not isinstance(result, Exception),
        )

        # If we didn't get a succesful parse, raise the appropriate error
        if samples is False:
            results_itr.seek(0)  # Reset to start of iterator
            errors = cast(list[Exception], list(results_itr))
            raise Sampler.SamplerError(sampler=samplers, error=errors)

        assert not isinstance(samples, (Exception, bool))
        return samples

    @overload
    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: None = None,
    ) -> Config:
        ...

    @overload
    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: int,
    ) -> list[Config]:
        ...

    def sample(
        self,
        space: Space,
        *,
        seed: Seed | None = None,
        n: int | None = None,
    ) -> Config | list[Config]:
        """Sample a configuration from the given space.

        Args:
            space: The space to sample from.
            seed: The seed to use for sampling.
            n: The number of configurations to sample.

        Returns:
            A single sample if `n` is None, otherwise a list of samples.
        """
        n = 1 if n is None else n
        samples = self._sample(space=space, n=n, seed=seed)
        return samples[0] if n == 1 else samples

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
