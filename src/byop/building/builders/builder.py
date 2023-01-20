"""Protocol defining what a Builder should be."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from result import Result

from byop.pipeline.pipeline import Pipeline
from byop.typing import BuiltPipeline


@runtime_checkable
class Builder(Protocol[BuiltPipeline]):
    """Attempts to build a usable object from a pipeline."""

    @classmethod
    def build(
        cls,
        pipeline: Pipeline,
    ) -> Result[BuiltPipeline, Exception]:
        """Build a pipeline into a usable object.

        Args:
            pipeline: The pipeline to build

        Returns:
            Result[BuiltPipeline, Exception]
        """
        ...
