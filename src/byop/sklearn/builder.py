"""Builds an sklearn.pipeline.Pipeline from a byop.pipeline.Pipeline."""
from __future__ import annotations

from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from typing_extensions import TypeAlias

from byop.pipeline.components import Component, Split
from byop.pipeline.pipeline import Pipeline
from byop.typing import Any, Name, Space

# split(
#   "name",
#   step("1", 1),
#   step("2", 1),
#   item=SklearnColumnTransformer,
#   config={
#     # These are associated with each path
#     "1": make_column_selector(dtype_include="int"),
#     "2": make_column_selector(dtype_include=object),
#
#     # These are known config options
#     # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
#     "remainder": "passthrough",
#     "sparse_threshold": 0.3,
#     "n_jobs": None,
#     "transformer_weights": None,
#     "verbose": False,
#     "verbose_feature_names_out": False,
#   }
# )

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
SklearnComponent: TypeAlias = Any | ColumnTransformer


def process_from(  # noqa: C901
    step: Component[str, SklearnComponent, Space] | Split[str, SklearnComponent, Space]
) -> Iterable[tuple[str, SklearnComponent]]:
    """Process a chain of steps into tuples of (name, component) for sklearn.

    Args:
        step: The head of the chain of steps to process

    Yields:
        tuple[str, SklearnComponent]: The name and component for sklearn
    """
    if isinstance(step, Component):
        yield (step.name, step.build())

    elif isinstance(step, Split):
        if step.item is None:
            raise ValueError(f"Can't handle split as it has no item attached: {step}")

        if step.item is ColumnTransformer:
            if any(path.name in COLUMN_TRANSFORMER_ARGS for path in step.paths):
                raise ValueError(
                    f"Can't handle step as it has a path with a name that matches"
                    f" a known ColumnTransformer argument: {step}"
                )

            # We passthrough if there's no config associated with the split as we
            # don't know what to pass to each possible path when the config is missing
            if step.config is None:
                yield (step.name, ColumnTransformer([], remainder="passthrough"))
            else:
                ct_config = {
                    k: v for k, v in step.config.items() if k in COLUMN_TRANSFORMER_ARGS
                }

                transformers: list = []
                for path in step.paths:
                    if path.name not in step.config:
                        transformers.append((path.name, "passthrough", None))
                    else:
                        assert isinstance(path, (Component, Split))
                        config = step.config[path.name]
                        steps = list(process_from(path))

                        sklearn_step: SklearnComponent
                        if len(steps) == 1:
                            sklearn_step = steps[0][1]
                        else:
                            sklearn_step = SklearnPipeline(steps)

                        transformers.append((path.name, sklearn_step, config))

                column_transformer = ColumnTransformer(transformers, **ct_config)
                yield (step.name, column_transformer)
        else:
            raise ValueError(
                f"Can't handle split as it has an unknown item: {step.item}"
                f" on step {step}."
            )

    if step.nxt is not None:
        assert isinstance(step.nxt, (Component, Split))
        yield from process_from(step.nxt)


def build(pipeline: Pipeline[str, Name]) -> SklearnPipeline:
    """Build a pipeline into a usable object.

    # TODO: SklearnPipeline has arguments not accessible to the outside caller.
    # We should expose these as well but I hesitate to do so with kwargs right
    # now.

    Args:
        pipeline: The pipeline to build

    Returns:
        SklearnPipeline
            The built pipeline
    """
    for step in pipeline.traverse():
        if not isinstance(step, (Component, Split)):
            msg = (
                f"Can't build pipeline with step {step}."
                " Only Components and Splits are supported."
            )
            raise ValueError(msg)

    assert isinstance(pipeline.head, (Component, Split))
    steps = list(process_from(pipeline.head))
    return SklearnPipeline(steps)
