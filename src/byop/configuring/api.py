"""Module for configuring a pipeline.

This means to take a `Pipeline` and configure it down to something
that can be eventually built. This giving each step the sub-configuration
from the larger chosen configuration and returning a new pipeline.
This includes trimming down choices to the chosen branch and removing
the spaces from the pipeline steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

from more_itertools import first_true, seekable
from result import Result, as_result

from byop.configuring.configurers import (
    DEFAULT_CONFIGURERS,
    ConfigurationError,
    Configurer,
)
from byop.pipeline.pipeline import Pipeline
from byop.typing import Config, Key, Name

if TYPE_CHECKING:
    import ConfigSpace


def configure(
    pipeline: Pipeline[Key, Name],
    config: Config | ConfigSpace.Configuration,
    *,
    configurer: (
        Literal["auto"]
        | Configurer
        | Callable[[Pipeline[Key, Name], Config], Pipeline[Key, Name]]
    ) = "auto",
) -> Pipeline[Key, Name]:
    """Configure a pipeline with a given config.

    Args:
        pipeline: The pipeline to configure.
        config: The configuration to use.
        configurer: The configurer to use. If "auto", will use the first
            configurer that can handle the config.

    Returns:
        The configured pipeline.
    """
    results: seekable[Result[Pipeline[Key, Name], ConfigurationError | Exception]]
    configurers: list[Any]

    if configurer is None:
        configurers = DEFAULT_CONFIGURERS
        results = seekable(
            configurer._configure(pipeline, config)
            for configurer in configurers
            if configurer.supports(pipeline, config)
        )

    elif isinstance(configurer, Configurer):
        configurers = [configurer]
        results = seekable([configurer._configure(pipeline, config)])

    elif callable(configurer):
        configurers = [configurer]
        safe_configurer = as_result(Exception)(configurer)
        results = seekable([safe_configurer(pipeline, config)])
    else:
        raise NotImplementedError(f"Configurer {configurer} not supported")

    selected_configuration = first_true(
        results,
        default=None,
        pred=lambda r: r.is_ok(),
    )

    if selected_configuration is None:
        results.seek(0)
        errors = [r.unwrap_err() for r in results]
        configuration_errors = [
            error for error in errors if isinstance(error, ConfigurationError)
        ]
        others = [
            error for error in errors if not isinstance(error, ConfigurationError)
        ]
        raise ValueError(
            f"Could not configure pipeline with the configurers, "
            f"\n{configurers=}"
            "\n"
            f" Parser errors:\n{configuration_errors=}"
            f" Other errors:\n{others=}"
        )

    assert selected_configuration.is_ok()
    return selected_configuration.unwrap()
