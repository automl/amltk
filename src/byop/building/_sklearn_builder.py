"""Build a pipeline into an [`sklearn.pipeline.Pipeline`][sklearn.pipeline.Pipeline]."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline as SklearnPipeline

    from byop.pipeline import Pipeline


def sklearn_builder(pipeline: Pipeline) -> SklearnPipeline:
    """Build a pipeline into a usable object.

    Args:
        pipeline: The pipeline to build

    Returns:
        The built sklearn pipeline
    """
    try:
        from byop.sklearn.builder import build

        return build(pipeline)
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "The sklearn builder requires the sklearn package to be installed. "
            "Please install it using `pip install sklearn`.",
        ) from e
