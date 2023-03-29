"""The parser to parse a configspace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

from byop.pipeline import Pipeline
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
        from byop.configspace.parser import generate_configspace

        return generate_configspace(pipeline, seed)
    except ModuleNotFoundError as e:
        errmsg = "Could not succesfully import ConfigSpace. Is it installed?"
        raise ImportError(errmsg) from e
