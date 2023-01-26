"""A parser for a pipeline to convert hyperparameters to a Space."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from byop.typing import Seed, Space

if TYPE_CHECKING:
    from result import Result

    from byop.pipeline import Pipeline


class ParseError(Exception):
    """Error raised when parsing fails."""


@runtime_checkable
class SpaceParser(Protocol[Space]):
    """Attempts to parse a pipeline into a space."""

    @classmethod
    def parse(cls, pipeline: Pipeline, seed: Seed | None = None) -> Space:
        """Create a new space from a pipeline."""
        result = cls._parse(pipeline, seed)

        if result.is_err():
            raise result.unwrap_err()

        return result.unwrap()

    @classmethod
    def _parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,
    ) -> Result[Space, ParseError | Exception]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed: The seed to use for the space generated

        Returns:
            Result[Space, ParseError]
        """
        ...

    @classmethod
    def supports(cls, t: type | Any) -> bool:
        """Whether this parser can parse a given Space."""
        ...
