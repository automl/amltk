"""Builds an sklearn.pipeline.Pipeline from a amltk.pipeline.Pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, TypeVar, Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

from amltk.pipeline.components import Component, Group, Split

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from amltk.pipeline.pipeline import Pipeline
    from amltk.pipeline.step import Step

COLUMN_TRANSFORMER_ARGS = [
    "remainder",
    "sparse_threshold",
    "n_jobs",
    "transformer_weights",
    "verbose",
    "verbose_feature_names_out",
]

# TODO: We can make this more explicit with typing out sklearn types.
# However sklearn operates in a bit more of a general level so it would
# require creating protocols to type this properly and work with sklearn's
# duck-typing.
SklearnItem: TypeAlias = Union[Any, ColumnTransformer]
SklearnPipelineT = TypeVar("SklearnPipelineT", bound=SklearnPipeline)


def process_component(
    step: Component[SklearnItem, Any],
) -> Iterable[tuple[str, SklearnItem]]:
    """Process a single step into a tuple of (name, component) for sklearn.

    Args:
        step: The step to process

    Returns:
        tuple[str, SklearnComponent]: The name and component for sklearn
    """
    yield (str(step.name), step.build())

    if step.nxt is not None:
        yield from process_from(step.nxt)


def process_group(step: Group[Any]) -> Iterable[tuple[str, SklearnItem]]:
    """Process a single group into a tuple of (name, component) for sklearn.

    !!! warning

        Only works for groups with a single item.

    Args:
        step: The step to process

    Returns:
        tuple[str, SklearnComponent]: The name and component for sklearn
    """
    if len(step) > 1:
        raise ValueError(
            f"Can't handle groups with more than 1 item: {step}."
            "\nCurrently they are simply removed and replaced with their one item."
            " If you inteded some other functionality with inclduing more than"
            " one item in a group, please raise a ticket or implement your own"
            " builder.",
        )

    single_path = step.paths[0]
    yield from process_from(single_path)

    if step.nxt is not None:
        yield from process_from(step.nxt)


def process_split(  # noqa: C901
    split: Split[ColumnTransformer, Any],
) -> Iterable[tuple[str, SklearnItem]]:
    """Process a single split into a tuple of (name, component) for sklearn.

    Args:
        split: The step to process

    Returns:
        tuple[str, SklearnComponent]: The name and component for sklearn
    """
    if split.item is None:
        raise NotImplementedError(
            f"Can't handle split as it has no item attached: {split}.",
            " Sklearn builder requires all splits to have a ColumnTransformer",
            " as the item.",
        )

    if isinstance(split.item, type) and not issubclass(split.item, ColumnTransformer):
        raise NotImplementedError(
            f"Can't handle split as it has a ColumnTransformer as the item: {split}.",
            " Sklearn builder requires all splits to have a subclass ColumnTransformer",
            " as the item.",
        )

    if split.config is None:
        raise NotImplementedError(
            f"Can't handle split as it has no config attached: {split}.",
            " Sklearn builder requires all splits to have a config to tell",
            " the ColumnTransformer how to operate.",
        )

    if any(path.name in COLUMN_TRANSFORMER_ARGS for path in split.paths):
        raise ValueError(
            f"Can't handle step as it has a path with a name that matches"
            f" a known ColumnTransformer argument: {split}",
        )

    path_names = {path.name for path in split.paths}

    # NOTE: If the path was previously under a `Choice`, then the choices name
    # will be in the config while the selected choice's name will not. We add
    # them here so they're added to the config and later we will remove the unsued
    # one.
    path_names.update(
        {path.old_parent for path in split.paths if path.old_parent is not None},
    )

    # Get the config values for the column transformer, and the paths
    ct_config = {k: v for k, v in split.config.items() if k in COLUMN_TRANSFORMER_ARGS}
    ct_paths = {k: v for k, v in split.config.items() if k in path_names}

    # ... if theirs any other values in the config that isn't these, raise an error
    if any(split.config.keys() - ct_config.keys() - ct_paths.keys()):
        raise ValueError(
            "Can't handle split as it has a config with keys that aren't"
            " ColumnTransformer arguments or paths"
            "\nPlease ensure that all keys in the config are either ColumnTransformer"
            " arguments or paths.\n"
            f"\nSplit '{split.name}': {split.config}"
            f"\nColumnTransformer arguments: {COLUMN_TRANSFORMER_ARGS}"
            f"\nPaths: {path_names}"
            f"\nPath config: {ct_paths}"
            f"\n{split.paths}"
            "\n",
        )

    for path in split.paths:
        if path.name not in ct_paths and path.old_parent in ct_paths:
            ct_paths[path.name] = ct_paths.pop(path.old_parent)

    transformers: list = []
    for path in split.paths:
        if path.name not in ct_paths:
            raise ValueError(
                f"Can't handle split {split.name=} as it has a path {path.name=}"
                " with no config associated with it."
                "\nPlease ensure that all paths have a config associated with them."
                f"\nSplit '{split.name}': {ct_paths}",
            )

        assert isinstance(path, (Component, Split, Group))
        steps = list(process_from(path))

        sklearn_step: SklearnItem

        sklearn_step = steps[0][1] if len(steps) == 1 else SklearnPipeline(steps)

        split_config = ct_paths[path.name]

        split_item = (path.name, sklearn_step, split_config)
        transformers.append(split_item)

    column_transformer_cls = split.item
    column_transformer = column_transformer_cls(transformers, **ct_config)
    yield (split.name, column_transformer)

    if split.nxt is not None:
        yield from process_from(split.nxt)


def process_from(step: Step) -> Iterable[tuple[str, SklearnItem]]:
    """Process a chain of steps into tuples of (name, component) for sklearn.

    Args:
        step: The head of the chain of steps to process

    Yields:
        tuple[str, SklearnComponent]: The name and component for sklearn
    """
    if isinstance(step, Split):
        yield from process_split(step)
    elif isinstance(step, Group):
        yield from process_group(step)
    elif isinstance(step, Component):
        yield from process_component(step)
    else:
        raise NotImplementedError(f"Can't handle step: {step}")


def build(
    pipeline: Pipeline,
    pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
    **pipeline_kwargs: Any,
) -> SklearnPipelineT:
    """Build a pipeline into a usable object.

    Args:
        pipeline: The pipeline to build
        pipeline_type: The type of pipeline to build. Defaults to the standard
            sklearn pipeline but can be any deritiative of that, i.e. imblearn's
            pipeline.
        **pipeline_kwargs: The kwargs to pass to the pipeline_type.

    Returns:
        The built pipeline
    """
    pipeline_kwargs = pipeline_kwargs or {}

    for step in pipeline.traverse():
        if not isinstance(step, (Component, Group, Split)):
            msg = (
                f"Can't build pipeline with step {step}."
                " Only Components and Splits are supported."
            )
            raise ValueError(msg)

    assert isinstance(pipeline.head, (Component, Split, Group))
    steps = list(process_from(pipeline.head))
    return pipeline_type(steps, **pipeline_kwargs)  # type: ignore
