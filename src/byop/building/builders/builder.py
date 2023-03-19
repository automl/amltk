"""Protocol defining what a Builder should be."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from result import Result

from byop.pipeline.pipeline import Pipeline
from byop.types import BuiltPipeline


class BuilderError(RuntimeError):
    """Base class for all builder errors."""

    ...


@runtime_checkable
class Builder(Protocol[BuiltPipeline]):
    """Attempts to build a usable object from a pipeline."""

    Error: ClassVar[type[BuilderError]] = BuilderError

    @classmethod
    def build(cls, pipeline: Pipeline) -> BuiltPipeline:
        """Build a pipeline in to a usable object."""
        result = cls._build(pipeline)

        if result.is_err():
            raise result.unwrap_err()

        return result.unwrap()

    @classmethod
    def _build(
        cls,
        pipeline: Pipeline,
    ) -> Result[BuiltPipeline, BuilderError | Exception]:
        """Build a pipeline into a usable object.

        Args:
            pipeline: The pipeline to build

        Returns:
            Result[BuiltPipeline, BuilderError | Exception]
        """
        ...
