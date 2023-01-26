"""A parser for a pipeline which contains no hyperparameters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from result import Err, Ok, Result

from byop.parsing.space_parsers.space_parser import ParseError, SpaceParser
from byop.pipeline.components import Choice, Component, Split, Step

if TYPE_CHECKING:
    from byop.pipeline import Pipeline
    from byop.typing import Seed


EMPTY_SPACE_INDICATORS: tuple[Any, ...] = (None, {}, [], ())


def space_required(step: Step | Component | Choice | Split) -> bool:
    """Whether a step requires a space."""
    # Choices always require a hyperparameter in a space of a pipeline as we
    # need to know which component to choose.
    if isinstance(step, Choice):
        return True

    return (
        isinstance(step, (Component, Split))
        and step.space not in EMPTY_SPACE_INDICATORS
    )


class NoSpaceParser(SpaceParser[None]):
    """Parser for when having `None` for a Space is valid.

    This is only the case in which no space is present for any component.
    """

    @classmethod
    def _parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,  # noqa: ARG003 # pyright: ignore
    ) -> Result[None, ParseError]:
        """Parsers the pipeline to see if it has no Space asscoaited with it."""
        pipeline.traverse()
        if any(space_required(step) for step in pipeline.traverse()):
            msg = (
                "Pipeline contains a step(s) which has either:"
                "\n * A space which is not a `None`."
                "\n * A `Choice` which requires a space to be formed."
                "\nThese are not supported by the NoSpaceParser."
            )
            return Err(ParseError(msg))

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
