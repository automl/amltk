"""Module for configuring a pipeline.

This means to take a `Pipeline` and configure it down to something
that can be eventually built. This giving each step the sub-configuration
from the larger chosen configuration and returning a new pipeline.
This includes trimming down choices to the chosen branch and removing
the spaces from the pipeline steps.
"""
from __future__ import annotations

from typing import Callable
from uuid import uuid4

from more_itertools import first_true, seekable

from byop.configuring._mapping_configurers import str_mapping_configurer
from byop.exceptions import attach_traceback
from byop.pipeline.pipeline import Pipeline
from byop.types import Config


class ConfiguringError(Exception):
    """Error when a pipeline could not be configured."""


def configure(
    pipeline: Pipeline,
    config: Config,
    *,
    rename: bool | str = False,
) -> Pipeline:
    """Configure a pipeline with a given config.

    Args:
        pipeline: The pipeline to configure.
        config: The configuration to use.
        rename: Whether to rename the pipeline. Defaults to `False`.
            * If `True`, the pipeline will be renamed using a random uuid
            * If a Name is provided, the pipeline will be renamed to that name

    Returns:
        The configured pipeline.
    """
    configurers = [str_mapping_configurer]

    def _configure(
        _configurer: Callable[[Pipeline, Config], Pipeline],
        _pipeline: Pipeline,
        _config: Config,
    ) -> Pipeline | Exception:
        try:
            return _configurer(_pipeline, _config)
        except Exception as e:  # noqa: BLE001
            return attach_traceback(e)

    itr = (_configure(_configurer, pipeline, config) for _configurer in configurers)
    results = seekable(itr)
    selected_configuration = first_true(
        results,
        default=None,
        pred=lambda r: not isinstance(r, Exception),
    )

    if selected_configuration is None:
        results.seek(0)
        results.seek(0)  # Reset to start of the iterator
        msgs = "\n".join(
            f"{configurer}: {err}" for configurer, err in zip(configurers, results)
        )
        raise ConfiguringError(
            f"Could not parse pipeline with any of the parser:\n{msgs}"
        )

    assert not isinstance(selected_configuration, Exception)

    # HACK: This is a hack to get around the fact that the pipeline
    # is frozen but we need to rename it.
    if rename is not False:
        new_name = str(uuid4()) if rename is True else rename
        object.__setattr__(selected_configuration, "name", new_name)

    return selected_configuration
