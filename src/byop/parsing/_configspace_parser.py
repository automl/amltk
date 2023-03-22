"""The parser to parse a configspace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

from byop.pipeline import Pipeline
from byop.pipeline.components import Component, Split
from byop.types import Seed

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


def configspace_parser(
    pipeline: Pipeline,
    seed: Seed | None = None,
) -> ConfigurationSpace:
    """Parse a configspace from a pipeline.

    Args:
        pipeline: The pipeline to parse the configspace from
        seed: The seed to use for the configspace

    Returns:
        The parsed configspace
    """
    try:
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import Hyperparameter

        from byop.configspace.parser import generate_configspace

        searchables = (
            s
            for s in pipeline.traverse()
            if isinstance(s, (Component, Split)) and s.space is not None
        )
        eligble_types = (ConfigurationSpace, dict, Hyperparameter)
        ineligibile = [s for s in searchables if not isinstance(s.space, eligble_types)]

        if any(ineligibile):
            errmsg = (
                "Pipeline contains a step(s) which has a space which is not a "
                f" ConfigSpace, dict or Hyperparameter. {ineligibile=}"
            )
            raise ValueError(errmsg)

        return generate_configspace(pipeline, seed)
    except ModuleNotFoundError as e:
        errmsg = "Could not succesfully import ConfigSpace. Is it installed?"
        raise ImportError(errmsg) from e
