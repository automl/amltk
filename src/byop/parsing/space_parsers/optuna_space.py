"""The parser to parse a configspace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from result import Err, Ok, Result

from byop.parsing.space_parsers.space_parser import ParseError, SpaceParser
from byop.pipeline.components import Component, Split

if TYPE_CHECKING:
    from byop.optuna_space.space_parsing import OPTUNA_SEARCH_SPACE
    from byop.pipeline import Pipeline
    from byop.types import Seed


class OptunaSpaceParser(SpaceParser["OPTUNA_SEARCH_SPACE"]):
    """Attempts to parse a pipeline into a ConfigSpace."""

    @classmethod
    def _parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,  # noqa[ARG003]
    ) -> Result[OPTUNA_SEARCH_SPACE, ParseError | Exception]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed: The seed to use for the space generated.

        Returns:
            Result[Space, ParseError]
        """
        try:
            from byop.optuna_space.space_parsing import generate_optuna_search_space

            searchables = (
                s
                for s in pipeline.traverse()
                if isinstance(s, (Component, Split)) and s.space is not None
            )
            eligble_types = dict
            ineligibile = [
                s for s in searchables if not isinstance(s.space, eligble_types)
            ]

            if any(ineligibile):
                errmsg = (
                    "Pipeline contains a step(s) which has a space which is not a "
                    f" dict. {ineligibile=}"
                )
                e = SpaceParser.Error(errmsg)
                return Err(e)

            # 2. Then we try generate the search space
            return Ok(generate_optuna_search_space(pipeline))
        except ModuleNotFoundError:
            errmsg = "Could not succesfully import optuna. Is it installed?"
            e = SpaceParser.Error(errmsg)
            return Err(e)
        except Exception as e:  # noqa: BLE001
            return Err(e)

    @classmethod
    def supports(cls, t: type | Any) -> bool:
        """Whether this parser can parse a given space type."""
        return t is dict or isinstance(t, dict)
