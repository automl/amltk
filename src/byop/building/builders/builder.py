"""Protocol defining what a Builder should be."""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable

from result import Result

from byop.pipeline.pipeline import Pipeline

B = TypeVar("B", covariant=True)


class BuilderError(RuntimeError):
    """Base class for all builder errors."""

    ...


@runtime_checkable
class Builder(Protocol[B]):
    """Attempts to build a usable object from a pipeline."""

    Error: ClassVar[type[BuilderError]] = BuilderError

    @classmethod
    def build(cls, pipeline: Pipeline) -> B:
        """Build a pipeline in to a usable object."""
        result = cls._build(pipeline)

        if result.is_err():
            raise result.unwrap_err()

        return result.unwrap()

    @classmethod
    def _build(
        cls,
        pipeline: Pipeline,
    ) -> Result[B, BuilderError | Exception]:
        """Build a pipeline into a usable object.

        Args:
            pipeline: The pipeline to build

        Returns:
            Result[B, BuilderError | Exception]
        """
        ...
