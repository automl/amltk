"""Space adapter.

This module provides the SpaceAdapter class, which is a parser and sampler
for a given Space type.
"""
from __future__ import annotations

import logging

from byop.pipeline.parser import Parser
from byop.pipeline.sampler import Sampler
from byop.types import Space

logger = logging.getLogger(__name__)


class SpaceAdapter(Parser[Space], Sampler[Space]):
    """Space adapter.

    This class is a parser and sampler for a given Space type.

    See Also:
        * [`Parser`][byop.pipeline.parser.Parser]
        * [`Sampler`][byop.pipeline.sampler.Sampler]
    """

    ...
