"""A configurer from some Configuration object to a Pipeline."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from result import Result

from byop.pipeline import Pipeline
from byop.typing import Config, Key, Name


class ConfigurationError(Exception):
    """An error that occurred during configuration."""


@runtime_checkable
class Configurer(Protocol[Key]):
    """Attempts to parse a pipeline into a space."""

    @classmethod
    def configure(
        cls,
        pipeline: Pipeline[Key, Name],
        config: Config,
    ) -> Pipeline[Key, Name]:
        """Configure a pipeline with a given config.

        Args:
            pipeline: The pipeline to configure.
            config: The configuration to use.

        Returns:
            Pipeline
        """
        result = cls._configure(pipeline, config)

        if result.is_err():
            raise result.unwrap_err()

        return result.unwrap()

    @classmethod
    def _configure(
        cls,
        pipeline: Pipeline[Key, Name],
        config: Config,
    ) -> Result[Pipeline[Key, Name], ConfigurationError | Exception]:
        """Takes a pipeline and a config to produce a configured pipeline.

        Args:
            pipeline: The pipeline to configure
            config: The config object to use

        Returns:
            Result[Pipeline, ConfigurationError | Exception]
        """
        ...

    @classmethod
    def supports(cls, pipeline: Pipeline, config: Config) -> bool:
        """Whether this configurer can use a given config on this pipeline."""
        ...