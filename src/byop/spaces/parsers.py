"""A module for parsers that can return a Space for a pipeline.

Each of these parsers can be used even if module requirements are not met
to allow for optional dependancies. The will each call into their respective
module to create the space if the module is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

from result import Err, Ok, Result
from typing_extensions import TypeAlias

from byop.pipeline import Pipeline
from byop.pipeline.components import Choice, Component, Split
from byop.spaces.configspace import generate_configspace
from byop.typing import Seed, Space

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


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


class ConfigSpaceParser(SpaceParser[ConfigurationSpace]):
    """Attempts to parse a pipeline into a ConfigSpace."""

    @classmethod
    def parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,
    ) -> Result[ConfigurationSpace, ModuleNotFoundError | ValueError]:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse
            seed: The seed to use for the space generated

        Returns:
            Result[Space, Exception]
        """
        try:
            from ConfigSpace import ConfigurationSpace

            # 1. First we do a quick traversal to see if everything is configspace
            # eligble
            searchables = (
                s
                for s in pipeline.traverse()
                if isinstance(s, (Component, Split)) and s.space is not None
            )
            # TODO: Enable individual hyperparametres, not requiing a space for step
            ineligibile = [
                s for s in searchables if not isinstance(s.space, ConfigurationSpace)
            ]

            if any(ineligibile):
                return Err(ValueError(f"Requires ConfigSpace space {ineligibile=}"))

            # 2. Then we generate the configspace
            return Ok(generate_configspace(pipeline, seed))
        except ModuleNotFoundError as e:
            return Err(e)

    @classmethod
    def supports(cls, t: type | Any) -> bool:
        """Whether this parser can parse a given space type."""
        # TODO: For the moment this is only itself but we could most likely
        # convert a neps space into a ConfigSpace
        try:
            from ConfigSpace import ConfigurationSpace

            return t is ConfigurationSpace or isinstance(t, ConfigurationSpace)
        except ModuleNotFoundError:
            return False


class NoSpaceParser(SpaceParser[None]):
    """Parser for when having `None` for a Space is valid.

    This is only the case in which no space is present for any component.
    """

    @classmethod
    def parse(
        cls,
        pipeline: Pipeline,
        seed: Seed | None = None,  # pyright: ignore
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
        t: type | Any,  # pyright: ignore
    ) -> Literal[True]:
        """Whether this parser can parse a given space type.

        Seeing as pipelines can always have no searchspaces defined, this
        always returns true
        """
        return True


ParserFunction: TypeAlias = Callable[
    [Pipeline, Optional[Seed]],
    Result[Space, Exception],
]


DEFAULT_PARSERS = [NoSpaceParser, ConfigSpaceParser]
