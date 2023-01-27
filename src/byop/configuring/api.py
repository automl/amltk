"""Module for configuring a pipeline.

This means to take a `Pipeline` and configure it down to something
that can be eventually built. This giving each step the sub-configuration
from the larger chosen configuration and returning a new pipeline.
This includes trimming down choices to the chosen branch and removing
the spaces from the pipeline steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping
from uuid import uuid4

from more_itertools import first, first_true, seekable
from result import Result, as_result

from byop.configuring.configurers import (
    DEFAULT_CONFIGURERS,
    ConfigSpaceConfigurer,
    ConfigurationError,
    Configurer,
    HeirarchicalStrConfigurer,
)
from byop.functional import reposition
from byop.pipeline.pipeline import Pipeline
from byop.typing import Config, Key, Name, safe_isinstance

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
    rename: bool | Name = False,
) -> Pipeline[Key, Name]:
    """Configure a pipeline with a given config.

    Args:
        pipeline: The pipeline to configure.
        config: The configuration to use.
        configurer: The configurer to use. If "auto", will use the first
            configurer that can handle the config.
        rename: Whether to rename the pipeline. Defaults to `False`.
            * If `True`, the pipeline will be renamed using a random uuid
            * If a Name is provided, the pipeline will be renamed to that name

    Returns:
        The configured pipeline.
    """
    results: seekable[Result[Pipeline[Key, Name], ConfigurationError | Exception]]
    configurers: list[Any]

    if configurer == "auto":
        configurers = DEFAULT_CONFIGURERS

        # If we can infer something about the config, prioritize that.
        if safe_isinstance(config, "Configuration"):
            configurers = reposition(configurers, [ConfigSpaceConfigurer, ...])

        elif isinstance(config, Mapping) and isinstance(
            first(config.keys(), default=None), str
        ):
            configurers = reposition(configurers, [HeirarchicalStrConfigurer, ...])

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
        errors = [(c, r.unwrap_err()) for c, r in zip(configurers, results)]
        okay_errors = [(c, e) for c, e in errors if isinstance(e, ConfigurationError)]
        others = [(c, e) for c, e in errors if not isinstance(e, ConfigurationError)]
        raise ValueError(
            f"Could not configure pipeline with the configurers, "
            f"\n{configurers=}"
            "\n"
            f" Configuration errors:\n{okay_errors=}"
            f" Other errors:\n{others=}"
        )

    assert selected_configuration.is_ok()
    pipeline = selected_configuration.unwrap()

    # HACK: This is a hack to get around the fact that the pipeline
    # is frozen but we need to rename it.
    if rename is not False:
        new_name = str(uuid4()) if rename is True else rename
        object.__setattr__(pipeline, "name", new_name)

    return pipeline
