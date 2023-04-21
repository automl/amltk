"""Space adapter."""
from __future__ import annotations

import logging

from byop.pipeline.parser import Parser
from byop.pipeline.sampler import Sampler
from byop.types import Space

logger = logging.getLogger(__name__)


class SpaceAdapter(Parser[Space], Sampler[Space]):
    """Space adapter.

    This interfaces combines the utility to parse and sample from a given
    type of Space.
    It is a combination of the [`Parser`][byop.pipeline.parser.Parser] and
    [`Sampler`][byop.pipeline.sampler.Sampler] interfaces, such that
    we can perform operations on a Space without knowing its type.

    To implement a new SpaceAdapter, you must implement the methods
    described in the [`Parser`][byop.pipeline.parser.Parser] and
    [`Sampler`][byop.pipeline.sampler.Sampler] interfaces.

    !!! example "Example Adapaters"

        We have integrated adapters for the following libraries which
        you can use as full reference guide.

        * [`OptunaSpaceAdapter`][byop.optuna.space.OptunaSpaceAdapter] for
            [Optuna](https://optuna.org/)
        * [`ConfigSpaceAdapter`][byop.configspace.space.ConfigSpaceAdapter]
            for [ConfigSpace](https://automl.github.io/ConfigSpace/master/)

    """

    @classmethod
    def default_adapters(cls) -> list[SpaceAdapter]:
        """Get the default adapters.

        Returns:
            A list of default adapters.
        """
        adapters: list[SpaceAdapter] = []
        try:
            from byop.optuna.space import OptunaSpaceAdapter

            adapters.append(OptunaSpaceAdapter())
        except ImportError:
            logger.debug("Optuna not installed, skipping adapter")

        try:
            from byop.configspace.space import ConfigSpaceAdapter

            adapters.append(ConfigSpaceAdapter())
        except ImportError:
            logger.debug("ConfigSpace not installed, skipping adapter")

        return adapters
