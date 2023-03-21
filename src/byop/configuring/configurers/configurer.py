"""A configurer from some Configuration object to a Pipeline."""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from result import Result

from byop.pipeline import Pipeline
from byop.types import Config


class ConfigurationError(Exception):
    """An error that occurred during configuration."""


class Configurer:
    """Attempts to parse a pipeline into a space."""

    Error: ClassVar[type[ConfigurationError]] = ConfigurationError

    @classmethod
    def configure(cls, pipeline: Pipeline, config: Config) -> Pipeline:
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
    @abstractmethod
    def _configure(
        cls,
        pipeline: Pipeline,
        config: Config,
    ) -> Result[Pipeline, ConfigurationError | Exception]:
        """Takes a pipeline and a config to produce a configured pipeline.

        Args:
            pipeline: The pipeline to configure
            config: The config object to use

        Returns:
            Result[Pipeline, ConfigurationError | Exception]
        """
        ...

    @classmethod
    @abstractmethod
    def supports(cls, pipeline: Pipeline, config: Config) -> bool:
        """Whether this configurer can use a given config on this pipeline."""
        ...
