"""Parser."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    Mapping,
    Sequence,
    cast,
    overload,
)

from more_itertools import first_true, seekable

from byop.exceptions import safe_map
from byop.pipeline.components import Choice, Group, Step
from byop.pipeline.pipeline import Pipeline
from byop.types import Seed, Space

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Error for when a Parser fails to parse a Pipeline."""

    @overload
    def __init__(self, parser: Parser, error: Exception):
        ...

    @overload
    def __init__(self, parser: list[Parser], error: list[Exception]):
        ...

    def __init__(
        self,
        parser: Parser | list[Parser],
        error: Exception | list[Exception],
    ):
        """Create a new parser error.

        Args:
            parser: The parser(s) that failed.
            error: The error(s) that was raised.

        Raises:
            ValueError: If parser is a list, exception must be a list of the
                same length.
        """
        if isinstance(parser, list) and (
            not (isinstance(error, list) and len(parser) == len(error))
        ):
            raise ValueError(
                "If parser is a list, `error` must be a list of the same length."
                f"Got {parser=} and {error=} .",
            )

        self.parser = parser
        self.error = error

    def __str__(self) -> str:
        if isinstance(self.parser, list):
            msg = "\n\n".join(
                f"Failed to parse with {p}:" + "\n " + f"{e.__class__.__name__}: {e}"
                for p, e in zip(self.parser, self.error)  # type: ignore
            )
        else:
            msg = (
                f"Failed to parse with {self.parser}:"
                + "\n"
                + f"{self.error.__class__.__name__}: {self.error}"
            )

        return msg


class Parser(ABC, Generic[Space]):
    """A parser to parse a Pipeline/Step's `search_space` into a Space.

    This class is a parser for a given Space type, providing functionality
    for the parsing algothim to run on a given search space. To implement
    a parser for a new search space, you must implement the abstract methods
    in this class.

    !!! example "Abstract Methods"

        * [`parse_space`][byop.pipeline.parser.Parser.parse_space]:
            Parse a search space into a space.
        * [`empty`][byop.pipeline.parser.Parser.empty]:
            Get an empty space.
        * [`insert`][byop.pipeline.parser.Parser.insert]:
            Insert a space into another space, with a possible prefix + delimiter.
        * [`set_seed`][byop.pipeline.parser.Parser.set_seed]:
            _(Optional)_ Set the seed of a space.

        !!! note

            If your space supports conditions, you must also implement:

            * [`condition`][byop.pipeline.parser.Parser.condition]:
                Condition a set of subspaces on their names, based on a hyperparameter
                with which takes on values with these names. Must be encoded as a Space.

        Please see the respective docstrings for more.

    See Also:
        * [`SpaceAdapter`][byop.pipeline.space.SpaceAdapter]
            Together with implementing the [`Sampler`][byop.pipeline.sampler.Sampler]
            interface, this class provides a complete adapter for a given search space.
    """

    ParserError: ClassVar[type[ParserError]] = ParserError
    """The error to raise when parsing fails."""

    @classmethod
    def default_parsers(cls) -> list[Parser]:
        """Get the default parsers."""
        parsers: list[Parser] = []

        try:
            from byop.configspace import ConfigSpaceAdapter

            parsers.append(ConfigSpaceAdapter())
        except ImportError:
            logger.debug("ConfigSpace not installed for parsing, skipping")

        try:
            from byop.optuna import OptunaSpaceAdapter

            parsers.append(OptunaSpaceAdapter())
        except ImportError:
            logger.debug("Optuna not installed for parsing, skipping")

        return parsers

    @classmethod
    def try_parse(
        cls,
        pipeline_or_step: Pipeline | Step,
        parser: type[Parser[Space]] | Parser[Space] | None = None,
        *,
        seed: Seed | None = None,
    ) -> Space:
        """Attempt to parse a pipeline with the default parsers.

        Args:
            pipeline_or_step: The pipeline or step to parse.
            parser: The parser to use. If `None`, will try all default parsers that
                are installed.
            seed: The seed to use for the parser.

        Returns:
            The parsed space.
        """
        if parser is None:
            parsers = cls.default_parsers()
        elif isinstance(parser, Parser):
            parsers = [parser]
        else:
            parsers = [parser()]

        if not any(parsers):
            raise RuntimeError(
                "Found no possible parser to use. Have you tried installing any of:"
                "\n* ConfigSpace"
                "\n* Optuna"
                "\nPlease see the integration documentation for more info, especially"
                "\nif using an optimizer which often requires a specific search space."
                "\nUsually just installing the optimizer will work.",
            )

        def _parse(_parser: Parser[Space]) -> Space:
            if pipeline_or_step is None:
                raise ValueError("Whut?")

            _parsed_space = _parser.parse(pipeline_or_step)
            if seed is not None:
                _parser.set_seed(_parsed_space, seed)
            return _parsed_space

        # Wrap in seekable so we don't evaluate all of them, only as
        # far as we need to get a succesful parse.
        results_itr = seekable(safe_map(_parse, parsers))

        is_result = lambda r: not (isinstance(r, tuple) and isinstance(r[0], Exception))
        # Progress the iterator until we get a successful parse
        parsed_space = first_true(results_itr, default=False, pred=is_result)

        # If we didn't get a succesful parse, raise the appropriate error
        if parsed_space is False:
            results_itr.seek(0)  # Reset to start of iterator
            errors = cast(list[Exception], list(results_itr))
            raise Parser.ParserError(parser=parsers, error=errors)

        assert not isinstance(parsed_space, (tuple, bool))
        return parsed_space

    def parse(self, step: Pipeline | Step | Group | Choice | Any) -> Space:
        """Parse a pipeline, step or something resembling a Space.

        Args:
            step: The pipeline or step to parse. If it is not
                a Pipeline object, it will be treated as a
                search space and attempt to be parsed as such.

        Returns:
            The space representing the pipeline or step.
        """
        if isinstance(step, Pipeline):
            return self.parse_pipeline(step)

        if isinstance(step, Choice):
            return self.parse_choice(step)

        if isinstance(step, Group):
            return self.parse_group(step)

        if isinstance(step, Step):
            return self.parse_step(step)

        return self.parse_space(step)

    def parse_pipeline(self, pipeline: Pipeline) -> Space:
        """Parse a pipeline into a space.

        Args:
            pipeline: The pipeline to parse.

        Returns:
            The space representing the pipeline. The pipeline will have no prefix
                while any modules attached to the pipeline will have the modules
                name as the prefix in the space
        """
        space = self.parse(pipeline.head)

        for module in pipeline.modules.values():
            prefix_delim = (module.name, ":") if isinstance(module, Pipeline) else None
            space = self.insert(space, self.parse(module), prefix_delim=prefix_delim)

        return space

    def parse_step(self, step: Step) -> Space:
        """Parse the space from a given step.

        Args:
            step: The step to parse.

        Returns:
            The space for this step.
        """
        space = self.empty()

        if step.search_space:
            _space = self.parse_space(step.search_space, step.config)
            space = self.insert(space, _space, prefix_delim=(step.name, ":"))

        if step.nxt is not None:
            _space = self.parse(step.nxt)
            space = self.insert(space, _space)

        return space

    def parse_group(self, step: Group) -> Space:
        """Parse the space from a given group.

        Args:
            step: The group to parse.

        Returns:
            The space for this group.
        """
        space = self.empty()

        if step.search_space:
            _space = self.parse_space(step.search_space, step.config)
            space = self.insert(space, _space, prefix_delim=(step.name, ":"))

        for path in step.paths:
            _space = self.parse(path)
            space = self.insert(space, _space, prefix_delim=(step.name, ":"))

        if step.nxt is not None:
            _space = self.parse(step.nxt)
            space = self.insert(space, _space)

        return space

    def parse_choice(self, step: Choice) -> Space:
        """Parse the space from a given choice.

        Note:
            This relies on the implementation of the `condition` method to
            condition the subspaces under the choice parameter. Please see
            the class docstring [here][byop.pipeline.parser.Parser] for more
            information.

        Args:
            step: The choice to parse.

        Returns:
            The space for this choice.
        """
        space = self.empty()

        if step.search_space:
            _space = self.parse_space(step.search_space, step.config)
            space = self.insert(space, _space, prefix_delim=(step.name, ":"))

        # Get all the subspaces for each choice
        subspaces = {path.name: self.parse(path) for path in step.paths}

        # Condition each subspace under some parameter "choice_name"
        conditioned_space = self.condition(
            choice_name=step.name,
            delim=":",
            spaces=subspaces,
            weights=step.weights,
        )

        space = self.insert(space, conditioned_space)

        if step.nxt is not None:
            _space = self.parse(step.nxt)
            space = self.insert(space, _space)

        return space

    def set_seed(self, space: Space, seed: Seed) -> Space:  # noqa: ARG002
        """Set the seed for the space.

        Overwrite if the can do something meaninfgul for the space.

        Args:
            space: The space to set the seed for.
            seed: The seed to set.

        Returns:
            The space with the seed set if applicable.
        """
        return space

    @abstractmethod
    def empty(self) -> Space:
        """Create an empty space.

        Returns:
            An empty space.
        """
        ...

    @abstractmethod
    def insert(
        self,
        space: Space,
        subspace: Space,
        *,
        prefix_delim: tuple[str, str] | None = None,
    ) -> Space:
        """Insert a subspace into a space.

        Args:
            space: The space to insert into.
            subspace: The subspace to insert.
            prefix_delim: The prefix, delimiter to use for the subspace.

        Returns:
            The original space with the subspace inserted.
        """
        ...

    def merge(self, *spaces: Space) -> Space:
        """Merge a list of spaces into a single space.


        ```python exec="true" source="material-block" result="python" title="Merging spaces"
        # Note, relies on ConfigSpace being installed `pip install ConfigSpace`
        from byop.configspace import ConfigSpaceAdapter

        adapter = ConfigSpaceAdapter()

        space_1 = adapter.parse({ "a": (1, 10) })
        space_2 = adapter.parse({ "b": (10.5, 100.5) })
        space_3 = adapter.parse({ "c": ["apple", "banana", "carrot"] })

        space = adapter.merge(space_1, space_2, space_3)

        print(space)
        ```

        Args:
            spaces: The spaces to merge.

        Returns:
            The merged space.
        """  # noqa: E501
        space = self.empty()

        for _space in spaces:
            space = self.insert(space, _space)

        return space

    @abstractmethod
    def parse_space(self, space: Any, config: Mapping[str, Any] | None = None) -> Space:
        """Parse a space from some object.

        Args:
            space: The space to parse.
            config: A possible set of concrete values to use for the space.
                If provided, the space should either set these values as constant
                or be excluded from the generated space.

        Returns:
            The parsed space.
        """
        ...

    @abstractmethod
    def condition(
        self,
        choice_name: str,
        delim: str,
        spaces: dict[str, Space],
        weights: Sequence[float] | None = None,
    ) -> Space:
        """Condition a set of spaces such that only one can be active.

        When sampling from the generated space. The choice name must be present
        along with the value it takes, which is any of the names of the choice paths.

        When a given choice is sampled, the corresponding subspace is sampled
        and none of the others.

        This must be encoded into the Space.

        If your space does not support conditionals, you can raise a
        an Error. If your space does support conditionals but not in this
        format, please raise an Issue!

        Args:
            choice_name: The name of the choice parameter.
            delim: The delimiter to use for the choice parameter.
            spaces: The spaces to condition. This is a mapping from the name
                of the choice to the space.
            weights: The weights to use for the choice parameter.
                If set and not possible, raise an Error.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
