from byop.assembly.space_parsers.configspace import ConfigSpaceParser
from byop.assembly.space_parsers.no_space import NoSpaceParser
from byop.assembly.space_parsers.parser import SpaceParser

# NOTE: Order here is important, as the first parser to support a space will be
# used.
# We defalt to the `NoSpace` parser, in which the space is None first because
# this is highly unlikely and escapes a issue of dealing with empty
# configuration space objects
DEFAULT_PARSERS = [NoSpaceParser, ConfigSpaceParser]

__all__ = ["ConfigSpaceParser", "NoSpaceParser", "SpaceParser", "DEFAULT_PARSERS"]
