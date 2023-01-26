"""A configurer from some Configuration object to a Pipeline."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from result import Result

from byop.pipeline import Pipeline
from byop.typing import Config


@runtime_checkable
class Configurer(Protocol):
    """Attempts to parse a pipeline into a space."""

    @classmethod
    def configure(
        cls,
        pipeline: Pipeline,
        config: Config,
    ) -> Result[Pipeline, Exception]:
        """Takes a pipeline and a config to produce a configured pipeline.

        Args:
            pipeline: The pipeline to configure
            config: The config object to use

        Returns:
            Result[Pipeline, Exception]
        """
        ...

    @classmethod
    def supports(cls, pipeline: Pipeline, config: Config) -> bool:
        """Whether this configurer can use a given config on this pipeline."""
        ...
