"""The parser to parse an OptunaSpace from a Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

from byop.pipeline import Pipeline
from byop.types import Seed

if TYPE_CHECKING:
    from byop.optuna import OptunaSearchSpace


def optuna_parser(
    pipeline: Pipeline,
    seed: Seed | None = None,  # noqa: ARG001
) -> OptunaSearchSpace:
    """Parse an OptunaSpace from a pipeline.

    Args:
        pipeline: The pipeline to parse the optuna space from
        seed: ignored

    Returns:
        The parsed optuna space
    """
    try:
        from byop.optuna.space import generate_optuna_search_space

        return generate_optuna_search_space(pipeline)
    except ModuleNotFoundError as e:
        errmsg = "Could not succesfully import optuna. Is it installed?"
        raise ImportError(errmsg) from e
