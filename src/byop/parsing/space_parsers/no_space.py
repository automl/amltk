"""A parser for a pipeline which contains no hyperparameters."""
from __future__ import annotations

from typing import Any, Literal

from result import Err, Ok, Result

from byop.parsing.space_parsers.space_parser import SpaceParser
from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split
from byop.typing import Seed


class NoSpaceParser(SpaceParser[None]):
    """Parser for when having `None` for a Space is valid.

    This is only the case in which no space is present for any component.
    """

    @classmethod
    def parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,  # noqa: ARG003 # pyright: ignore
    ) -> Result[None, Exception]:
        """Parsers the pipeline to see if it has no Space asscoaited with it."""
        if any(
            (isinstance(step, (Component, Split)) and step.space is not None)
            or isinstance(step, Choice)
            for step in pipeline.traverse()
        ):
            return Err(
                ValueError("Pipeline contains a step which has a space or is a choice")
            )

        return Ok(value=None)

    @classmethod
    def supports(
        cls,
        t: type | Any,  # noqa: ARG003 # pyright: ignore
    ) -> Literal[True]:
        """Whether this parser can parse a given space type.

        Seeing as pipelines can always have no searchspaces defined, this
        always returns true
        """
        return True
