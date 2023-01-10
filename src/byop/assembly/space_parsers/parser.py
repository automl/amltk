"""A parser for a pipeline to convert hyperparameters to a Space."""
from __future__ import annotations

from typing import Any, Protocol

from result import Result

from byop.pipeline import Pipeline
from byop.typing import Seed, Space


class SpaceParser(Protocol[Space]):
    """Attempts to parse a pipeline into a space."""

    @classmethod
    def parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,
    ) -> Result[Space, Exception]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed: The seed to use for the space generated

        Returns:
            Result[Space, Exception]
        """
        ...

    @classmethod
    def supports(cls, t: type | Any) -> bool:
        """Whether this parser can parse a given Space."""
        ...
