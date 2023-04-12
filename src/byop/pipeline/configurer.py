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
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, TypeVar

from more_itertools import first_true

from byop.pipeline.components import Choice, Split, Step
from byop.pipeline.pipeline import Pipeline

StepT = TypeVar("StepT", bound=Step)


def remove_prefix(key: str, prefix: str, delimiter: str) -> str:
    """Remove the prefix from the key, including the delimiter.

    Args:
        key: The key to remove the prefix from
        prefix: The prefix to remove
        delimiter: The delimiter to use to separate the names of the
            hierarchy.

    Returns:
        The key with the prefix removed
    """
    prefix_len = len(prefix) + len(delimiter)
    return key[prefix_len:]


def _validate_names(pipeline: Pipeline | Step, delimiter: str) -> None:
    """Recursively validate that the names of the steps in a pipeline
    do not contain the delimiter, including any modules attached to it.

    Args:
        pipeline: The pipeline to validate
        delimiter: The delimiter to use to separate the names of the
            hierarchy.
    """
    if isinstance(pipeline, Step):
        if delimiter in pipeline.name:
            raise ValueError(f"{delimiter=} is in step name: `{pipeline.name}`")
        if pipeline.nxt is not None:
            _validate_names(pipeline.nxt, delimiter=delimiter)

        # TODO: Handle choices, splits
        return

    for _, _, step in pipeline.walk():
        if delimiter in step.name:
            raise ValueError(f"{delimiter=} is in step name: `{step.name}`")

    for module_name, module in pipeline.modules.items():
        if delimiter in module_name:
            raise ValueError(f"{delimiter=} is in module name: `{module_name}`")
        try:
            _validate_names(module, delimiter=delimiter)
        except ValueError as e:
            raise ValueError(f"{module_name}: {e}") from e


def find_config_with_key(
    key: str,
    config: Mapping[str, Any],
    *,
    delimiter: str = ":",
    trim: bool = True,
) -> dict[str, Any]:
    """Select all entries in the config that start with the key and
    and optionally trim away the key and delimiter.

    Args:
        key: The key to select
        config: The config to select from
        delimiter: The delimiter to use to separate the names of the
            hierarchy. Defaults to ":".
        trim: Whether to trim the key and delimiter from the keys of
            the returned config. Defaults to True.

    Returns:
        The selected config
    """
    if trim:
        return {
            remove_prefix(k, key, delimiter): v
            for k, v in config.items()
            if k.startswith(key + delimiter)
        }

    return {k: v for k, v in config.items() if k.startswith(key + delimiter)}


def configure(
    pipeline: Pipeline,
    config: Mapping[str, Any],
    *,
    delimiter: str = ":",
    rename: str | bool = False,
) -> Pipeline:
    """Configure a pipeline using a mapping of strings to values.

    Args:
        pipeline: The pipeline to configure
        config: The config to use
        delimiter: The delimiter to use to separate the names of the
            hierarchy. Defaults to ":".
        rename: Whether to rename the pipeline. Defaults to `False`.
            * If `True`, the pipeline will be renamed using a random uuid
            * If a Name is provided, the pipeline will be renamed to that name

    Returns:
        Pipeline: The configured pipeline
    """
    _validate_names(pipeline, delimiter=delimiter)

    configured_pipeline_steps = _process(pipeline.head, config, delimiter=delimiter)

    pipeline_modules: list[Pipeline] = [
        m for m in pipeline.modules.values() if isinstance(m, Pipeline)
    ]
    step_modules: list[Step] = [
        m for m in pipeline.modules.values() if not isinstance(m, Pipeline)
    ]

    configured_pipeline_modules = [
        configure(
            pipeline=module,
            config=find_config_with_key(
                module.name,
                config,
                delimiter=delimiter,
                trim=True,
            ),
            delimiter=delimiter,
        )
        for module in pipeline_modules
    ]
    configured_step_modules = [
        Step.join(_process(m, config=config)) for m in step_modules
    ]

    if rename is False:
        _rename = pipeline.name
    elif rename is True:
        _rename = None
    else:
        _rename = rename

    return Pipeline.create(
        configured_pipeline_steps,
        modules=[*configured_step_modules, *configured_pipeline_modules],
        name=_rename,
    )


def _process(
    step: Step | Pipeline,
    config: Mapping[str, Any],
    *,
    splits: list[Split] | None = None,
    delimiter: str = ":",
) -> Iterator[Step]:
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
    if isinstance(step, Pipeline):
        return _process(step.head, config, delimiter=delimiter)

    segments = [s.name for s in (*splits, step)] if splits is not None else [step.name]
    step_key = delimiter.join(segments)

    def is_for_step(_key: str) -> bool:
        return _key.startswith(f"{step_key}{delimiter}")

    def is_for_path(_key: str, _paths: Iterable[Step]) -> bool:
        return any(
            _key.startswith(f"{step_key}{delimiter}{_path.name}") for _path in _paths
        )

    # Select the config for this step
    if isinstance(step, Choice):
        chosen_name = config.get(step_key, None)
        if chosen_name is None:
            raise ValueError(f"Choice {step_key=} not found in {config=}")

        chosen_path = first_true(step.paths, None, lambda p: (p.name == chosen_name))
        if chosen_path is None:
            raise ValueError(f"Choice {chosen_name=} not found in {step.paths}")

        new_splits: list[Split] = [*splits, step] if splits else [step]
        yield from _process(chosen_path, config, splits=new_splits, delimiter=delimiter)

    elif isinstance(step, Split):
        selected_config = {
            remove_prefix(k, step_key, delimiter): v
            for k, v in config.items()
            if is_for_step(k) and not is_for_path(k, step.paths)
        }
        predefined_config = step.config if step.config is not None else {}
        new_config = {**predefined_config, **selected_config}

        new_splits = [*splits, step] if splits else [step]
        paths = [
            Step.join(_process(path, config, splits=new_splits, delimiter=delimiter))
            for path in step.paths
        ]
        yield step.mutate(
            paths=paths,
            config=new_config if len(new_config) > 0 else None,
            search_space=None,
        )

    elif isinstance(step, Step):
        selected_config = {
            remove_prefix(k, step_key, delimiter): v
            for k, v in config.items()
            if is_for_step(k)
        }
        predefined_config = step.config if step.config is not None else {}
        new_config = {**predefined_config, **selected_config}
        yield step.mutate(
            config=new_config if len(new_config) > 0 else None,
            search_space=None,
        )

    else:
        raise ValueError(f"Unknown step type: {type(step)}")

    if step.nxt is not None:
        yield from _process(step.nxt, config, splits=splits, delimiter=delimiter)
