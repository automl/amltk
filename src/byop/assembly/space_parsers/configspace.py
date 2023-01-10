"""The parser to parse a configspace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from result import Err, Ok, Result

from byop.assembly.space_parsers.parser import SpaceParser
from byop.pipeline import Pipeline
from byop.pipeline.components import Component, Split
from byop.typing import Seed

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


class ConfigSpaceParser(SpaceParser["ConfigurationSpace"]):
    """Attempts to parse a pipeline into a ConfigSpace."""

    @classmethod
    def parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,
    ) -> Result[ConfigurationSpace, ModuleNotFoundError | ValueError]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed: The seed to use for the space generated

        Returns:
            Result[Space, Exception]
        """
        try:
            from ConfigSpace import ConfigurationSpace

            from byop.spaces.configspace import generate_configspace

            # 1. First we do a quick traversal to see if everything is configspace
            # eligble
            searchables = (
                s
                for s in pipeline.traverse()
                if isinstance(s, (Component, Split)) and s.space is not None
            )
            # TODO: Enable individual hyperparametres, not requiing a space for step
            ineligibile = [
                s for s in searchables if not isinstance(s.space, ConfigurationSpace)
            ]

            if any(ineligibile):
                return Err(ValueError(f"Requires ConfigSpace space {ineligibile=}"))

            # 2. Then we generate the configspace
            return Ok(generate_configspace(pipeline, seed))
        except ModuleNotFoundError as e:
            return Err(e)

    @classmethod
    def supports(cls, t: type | Any) -> bool:
        """Whether this parser can parse a given space type."""
        # TODO: For the moment this is only itself but we could most likely
        # convert a neps space into a ConfigSpace
        try:
            from ConfigSpace import ConfigurationSpace

            return t is ConfigurationSpace or isinstance(t, ConfigurationSpace)
        except ModuleNotFoundError:
            return False
