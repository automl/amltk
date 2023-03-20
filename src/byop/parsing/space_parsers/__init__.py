from __future__ import annotations

from byop.parsing.space_parsers.configspace import ConfigSpaceParser
from byop.parsing.space_parsers.no_space import NoSpaceParser
from byop.parsing.space_parsers.optuna_space import OptunaSpaceParser
from byop.parsing.space_parsers.space_parser import ParseError, SpaceParser

# NOTE: Order here is important, as the first parser to support a space will be
# used.
# We defalt to the `NoSpace` parser, in which the space is None first because
# this is highly unlikely and escapes a issue of dealing with empty space objects
# on a pipeline.
DEFAULT_PARSERS: list[type[SpaceParser]] = [
    NoSpaceParser,
    ConfigSpaceParser,
    OptunaSpaceParser,
]

__all__ = [
    "ConfigSpaceParser",
    "NoSpaceParser",
    "SpaceParser",
    "DEFAULT_PARSERS",
    "ParseError",
    "OptunaSpaceParser",
]
