"""Code for configuring a pipeline using a mapping of strings to values.

This expects a heirarchical mapping of strings to values.
For example, given a pipeline like so:

```python
from byop import Pipeline, step, split

pipeline = Pipeline.create(
    step("one", space={"v": [1, 2, 3]})
    split("split", 
        step("x", space={"v": [4, 5, 6]}),
        step("y", space={"v": [4, 5, 6]}),
    ),
    choice("choice", 
        step("a", space={"v": [4, 5, 6]}),
        step("b", space={"v": [4, 5, 6]}),
    )

)
```

Then you could configure a pipeline with the following config:

```python
config = {
    "one:v": 1,
    "split:x:v": 4,
    "split:y:v": 5,
    "choice": "a",
    "choice:a": 6,
}
```

The `delimiter` ":" is arbitrary but should not be contained in step
or hyperparameter names.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping

from more_itertools import first, first_true
from result import Err, Ok, Result

from byop.configuring.configurers.configurer import ConfigurationError, Configurer
from byop.pipeline.components import Choice, Component, Split, Step
from byop.pipeline.pipeline import Pipeline
from byop.typing import Config, Name


class HeirarchicalStrConfigurer(Configurer[str]):
    """A configurer that uses a mapping of strings to values."""

    @classmethod
    def _configure(
        cls,
        pipeline: Pipeline[str, Name],
        config: Mapping[str, Any],
        *,
        delimiter: str = ":",  # TODO: This could be a list of things to try
    ) -> Result[Pipeline[str, Name], ConfigurationError | Exception]:
        """Takes a pipeline and a config to produce a configured pipeline.

        Relies on there being a flat map structure in the config where the
        keys map to the names of the components in the pipeline.

        For nested pipelines, the delimiter is used to separate the names
        of the heriarchy.

        Args:
            pipeline: The pipeline to configure
            config: The config object to use
            delimiter: The delimiter to use to separate the names of the
                hierarchy.

        Returns:
            Result[Pipeline, Exception]
        """
        # Make sure we don't have the delimiter in a name of a step.
        for _, _, step in pipeline.walk():
            if delimiter in step.name:
                msg = f"Delimiter {delimiter} is in step name: `{step.name}`"
                return Err(ConfigurationError(msg))

        try:
            configured_steps = _process(pipeline.head, config, delimiter=delimiter)
            result = Pipeline.create(configured_steps, name=pipeline.name)
            return Ok(result)
        except (ConfigurationError, Exception) as e:
            return Err(e)

    @classmethod
    def supports(cls, pipeline: Pipeline, config: Config) -> bool:
        """Whether this configurer can use a given config on this pipeline."""
        if isinstance(config, Mapping):
            # We of course support an empty configuration, I think ...
            if len(config) == 0:
                return True

            first_key = first(config.keys())
            return isinstance(first_key, type(pipeline.head.name))

        return False


def _process(
    step: Step[str],
    config: Mapping[str, Any],
    *,
    splits: list[Split] | None = None,
    delimiter: str = ":",
) -> Iterator[Step[str]]:
    """Process a step, returning a new step with the config applied.

    Args:
        step: The step to process
        config: The config to apply
        splits: The splits that preceed this step
        delimiter: The delimiter to use to separate the names of the
            hierarchy. Defaults to ":".

    Returns:
        Step[str]: The new step with the config applied
    """
    segments = [step.name] if splits is None else [s.name for s in splits] + [step.name]

    prefix = delimiter.join(segments) + ":"
    prefix_len = len(prefix)

    if isinstance(step, Component):
        conf = {k[prefix_len:]: v for k, v in config.items() if k.startswith(prefix)}
        yield step.mutate(config=conf, space=None)

    elif isinstance(step, Choice):
        chosen_name = config.get(prefix[:-1], None)
        if chosen_name is not None:
            chosen = first_true(step.paths, None, lambda s: (s.name == chosen_name))
            if chosen is None:
                msg = f"Choice {chosen_name} not found in {step.paths}"
                raise ConfigurationError(msg)

            new_splits: list[Split] = [*splits, step] if splits else [step]
            yield from _process(chosen, config, splits=new_splits, delimiter=delimiter)

    elif isinstance(step, Split):
        new_splits = [*splits, step] if splits else [step]
        configured_paths = [
            Step.join(_process(st, config, splits=new_splits, delimiter=delimiter))
            for st in step.paths
        ]

        # The configuration for this split step is anything that:
        # * starts with the expected `prefix`, which includes this steps name
        # * Does not have a f"{prefix}{delimiter}{path}" in it,
        #   indicating that it is a config for a path in this split and not
        #   for this split itself.
        split_config = {
            k[prefix_len:]: v
            for k, v in config.items()
            if (
                k.startswith(prefix)
                and not any(
                    k.replace(prefix, "").startswith(path.name) for path in step.paths
                )
            )
        }

        if step.config is not None:
            split_config.update(step.config)

        yield step.mutate(
            paths=configured_paths,
            config=split_config if len(split_config) > 0 else None,
            space=None,
        )

    if step.nxt is not None:
        yield from _process(step.nxt, config, splits=splits, delimiter=delimiter)

    return
