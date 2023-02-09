"""The parser to parse a configspace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from result import Err, Ok, Result

from byop.parsing.space_parsers.space_parser import ParseError, SpaceParser
from byop.pipeline.components import Component, Split

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from byop.pipeline import Pipeline
    from byop.typing import Seed


class ConfigSpaceParser(SpaceParser["ConfigurationSpace"]):
    """Attempts to parse a pipeline into a ConfigSpace."""

    @classmethod
    def _parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,
    ) -> Result[ConfigurationSpace, ParseError | Exception]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed (optional): The seed to use for the space generated

        Returns:
            Result[Space, ParseError]
        """
        try:
            from ConfigSpace import ConfigurationSpace
            from ConfigSpace.hyperparameters import Hyperparameter

            from byop.configspace.space_parsing import generate_configspace

            searchables = (
                s
                for s in pipeline.traverse()
                if isinstance(s, (Component, Split)) and s.space is not None
            )
            eligble_types = (ConfigurationSpace, dict, Hyperparameter)
            ineligibile = [
                s for s in searchables if not isinstance(s.space, eligble_types)
            ]

            if any(ineligibile):
                errmsg = (
                    "Pipeline contains a step(s) which has a space which is not a "
                    f" ConfigSpace, dict or Hyperparameter. {ineligibile=}"
                )
                e = SpaceParser.Error(errmsg)
                return Err(e)

            # 2. Then we try generate the configspace
            return Ok(generate_configspace(pipeline, seed))
        except ModuleNotFoundError:
            errmsg = "Could not succesfully import ConfigSpace. Is it installed?"
            e = SpaceParser.Error(errmsg)
            return Err(e)
        except Exception as e:  # noqa: BLE001
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
