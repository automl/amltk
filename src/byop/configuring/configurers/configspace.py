"""Code for configuring a pipeline using a Configuration from ConfigSpace."""
from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from result import Result

from byop.configuring.configurers.configurer import ConfigurationError, Configurer
from byop.configuring.configurers.heirarchical_str import HeirarchicalStrConfigurer
from byop.pipeline.pipeline import Pipeline
from byop.types import Config, Name

if TYPE_CHECKING:
    from ConfigSpace import Configuration


class ConfigSpaceConfigurer(Configurer[str]):
    """A Configurer that uses a configure a pipeline."""

    @classmethod
    def _configure(
        cls,
        pipeline: Pipeline[str, Name],
        config: Configuration,
        *,
        delimiter: str = ":",  # TODO: This could be a list of things to try
    ) -> Result[Pipeline[str, Name], ConfigurationError | Exception]:
        """Takes a pipeline and a config to produce a configured pipeline.

        Relies on there being a flat map structure in the config where the
        keys map to the names of the components in the pipeline.

        For nested pipelines, the delimiter is used to separate the names
        of the heriarchy.

        Args:
            pipeline: The pipeline to configure
            config: The config object to use
            delimiter: The delimiter to use to separate the names of the
                hierarchy.

        Returns:
            Result[Pipeline, ConfigurationError]
        """
        # NOTE: As a ConfigSpace::Configuration acts as a mapping, we can just
        # default to this
        return HeirarchicalStrConfigurer._configure(
            pipeline,
            config,
            delimiter=delimiter,
        )

    @classmethod
    def supports(cls, pipeline: Pipeline, config: Config) -> bool:
        """Whether this configurer can use a given config on this pipeline."""
        with suppress(Exception):
            from ConfigSpace import Configuration

            if not isinstance(config, Configuration):
                return False

            return pipeline.head is not None and isinstance(pipeline.head.name, str)

        return False
