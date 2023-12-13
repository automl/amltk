"""The sklearn [`builder()`][amltk.pipeline.builders.sklearn.build], converts
a pipeline made of [`Node`][amltk.pipeline.Node]s into a sklearn
[`Pipeline`][sklearn.pipeline.Pipeline].

!!! tip "Requirements"

    This requires `sklearn` which can be installed with:

    ```bash
    pip install "amltk[scikit-learn]"

    # Or directly
    pip install scikit-learn
    ```

Each _kind_ of node corresponds to a different part of the end pipeline:

=== "`Fixed`"

    [`Fixed`][amltk.pipeline.Fixed] - The estimator will simply be cloned, allowing you
    to directly configure some object in a pipeline.

    ```python exec="true" source="material-block" html="true"
    from sklearn.ensemble import RandomForestClassifier
    from amltk.pipeline import Fixed

    est = Fixed(RandomForestClassifier(n_estimators=25))
    built_pipeline = est.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, built_pipeline)  # markdown-exec: hide
    ```

=== "`Component`"

    [`Component`][amltk.pipeline.Component] - The estimator will be built from the
    component's config. This is mostly useful to allow a space to be defined for
    the component.

    ```python exec="true" source="material-block" html="true"
    from sklearn.ensemble import RandomForestClassifier
    from amltk.pipeline import Component

    est = Component(RandomForestClassifier, space={"n_estimators": (10, 100)})

    # ... Likely get the configuration through an optimizer or sampling
    configured_est = est.configure({"n_estimators": 25})

    built_pipeline = configured_est.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, built_pipeline)  # markdown-exec: hide
    ```

=== "`Sequential`"

    [`Sequential`][amltk.pipeline.Sequential] - The sequential will be converted into a
    [`Pipeline`][sklearn.pipeline.Pipeline], building whatever nodes are contained
    within in.

    ```python exec="true" source="material-block" html="true"
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from amltk.pipeline import Component, Sequential

    pipeline = Sequential(
        PCA(n_components=3),
        Component(RandomForestClassifier, config={"n_estimators": 25})
    )
    built_pipeline = pipeline.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, built_pipeline)  # markdown-exec: hide
    ```

=== "`Split`"

    [`Split`][amltk.pipeline.Split] - The split will be converted into a
    [`ColumnTransformer`][sklearn.compose.ColumnTransformer], where each path
    and the data that should go through it is specified by the split's config.
    You can provide a `ColumnTransformer` directly as the item to the `Split`,
    or otherwise if left blank, it will default to the standard sklearn one.

    You can use a `Fixed` with the special keyword `"passthrough"` as you might normally
    do with a `ColumnTransformer`.

    By default, we provide two special keywords you can provide to a `Split`,
    namely `#!python "categorical"` and `#!python "numerical"`, which will
    automatically configure a `ColumnTransorfmer` to pass the appropraite
    columns of a data-frame to the given paths.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Split, Component
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    categorical_pipeline = [
        SimpleImputer(strategy="constant", fill_value="missing"),
        Component(
            OneHotEncoder,
            space={
                "min_frequency": (0.01, 0.1),
                "handle_unknown": ["ignore", "infrequent_if_exist"],
            },
            config={"drop": "first"},
        ),
    ]
    numerical_pipeline = [SimpleImputer(strategy="median"), StandardScaler()]

    split = Split(
        {
            "categorical": categorical_pipeline,
            "numerical": numerical_pipeline
        }
    )
    from amltk._doc import doc_print; doc_print(print, split)  # markdown-exec: hide
    ```

    You can manually specify the column selectors if you prefer.

    ```python
    split = Split(
        {
            "categories": categorical_pipeline,
            "numbers": numerical_pipeline,
        },
        config={
            "categories": make_column_selector(dtype_include=object),
            "numbers": make_column_selector(dtype_include=np.number),
        },
    )
    ```

=== "`Join`"

    [`Join`][amltk.pipeline.Join] - The join will be converted into a
    [`FeatureUnion`][sklearn.pipeline.FeatureUnion].

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Join, Component
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest

    join = Join(PCA(n_components=2), SelectKBest(k=3), name="my_feature_union")

    pipeline = join.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, pipeline)  # markdown-exec: hide
    ```

=== "`Choice`"

    [`Choice`][amltk.pipeline.Choice] - The estimator will be built from the chosen
    component's config. This is very similar to [`Component`][amltk.pipeline.Component].

    ```python exec="true" source="material-block" html="true"
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from amltk.pipeline import Choice

    # The choice here is usually provided during the `.configure()` step.
    estimator_choice = Choice(
        RandomForestClassifier(),
        MLPClassifier(),
        config={"__choice__": "RandomForestClassifier"}
    )

    built_pipeline = estimator_choice.build("sklearn")
    from amltk._doc import doc_print; doc_print(print, built_pipeline)  # markdown-exec: hide
    ```


"""  # noqa: E501
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypeVar

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline as SklearnPipeline,
)

from amltk.pipeline import (
    Choice,
    Component,
    Fixed,
    Join,
    Node,
    Searchable,
    Sequential,
    Split,
)

if TYPE_CHECKING:
    from typing import TypeAlias

COLUMN_TRANSFORMER_ARGS = [
    "remainder",
    "sparse_threshold",
    "n_jobs",
    "transformer_weights",
    "verbose",
    "verbose_feature_names_out",
]
FEATURE_UNION_ARGS = ["n_jobs", "transformer_weights", "verbose"]

# TODO: We can make this more explicit with typing out sklearn types.
# However sklearn operates in a bit more of a general level so it would
# require creating protocols to type this properly and work with sklearn's
# duck-typing.
SklearnItem: TypeAlias = Any | ColumnTransformer
SklearnPipelineT = TypeVar("SklearnPipelineT", bound=SklearnPipeline)


def _process_split(
    node: Split,
    pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
    **pipeline_kwargs: Any,
) -> tuple[str, ColumnTransformer]:
    if any(child.name in COLUMN_TRANSFORMER_ARGS for child in node.nodes):
        raise ValueError(
            f"Can't handle step as it has a path with a name that matches"
            f" a known ColumnTransformer argument: {node}",
        )

    config = dict(node.config) if node.config is not None else {}

    # Automatic categories/numerical config
    for keyword, column_selector_kwargs in (
        ("categorical", {"dtype_exclude": "number"}),
        ("numerical", {"dtype_include": "number"}),
    ):
        if any(child.name == keyword for child in node.nodes) and keyword not in config:
            config[keyword] = make_column_selector(**column_selector_kwargs)

    # Automatic numeric config
    if any(child.name not in config for child in node.nodes):
        raise ValueError(
            f"Can't handle split {node.name=} as some path has no config associated"
            " with it."
            "\nPlease ensure that all paths have a config associated with them."
            f"\n{config=}\n"
            f"children={[child.name for child in node.nodes]}\n",
        )

    match node.item:
        case None:
            col_transform_cls = ColumnTransformer
        case type() if issubclass(node.item, ColumnTransformer):
            col_transform_cls = node.item
        case _:
            raise ValueError(
                f"Can't handle: {node}.\n"
                " Requires all splits to have a subclass"
                " ColumnTransformer as the item, or None.",
            )

    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # list of (name, estimator, columns)
    transformers: list[tuple[str, Any, Any]] = []

    for child in node.nodes:
        child_steps = list(_iter_steps(child))
        match child_steps:
            case []:
                raise ValueError(f"Can't handle child of split.\n{child=}\n{node}")
            case [(name, sklearn_thing)]:
                transformers.append((name, sklearn_thing, config[child.name]))
            case list():
                sklearn_thing = pipeline_type(child_steps, **pipeline_kwargs)
                transformers.append((child.name, sklearn_thing, config[child.name]))

    return (node.name, col_transform_cls(transformers))


def _process_join(
    node: Join,
    pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
    **pipeline_kwargs: Any,
) -> tuple[str, FeatureUnion]:
    if any(child.name in FEATURE_UNION_ARGS for child in node.nodes):
        raise ValueError(
            f"Can't handle step as it has a path with a name that matches"
            f" a known FeatureUnion argument: {node}",
        )

    match node.item:
        case None:
            feature_union_cls = FeatureUnion
        case type() if issubclass(node.item, FeatureUnion):
            feature_union_cls = node.item
        case _:
            raise ValueError(
                f"Can't handle: {node}.\n"
                " Requires all splits to have a subclass"
                " ColumnTransformer as the item, or None.",
            )

    # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    # list of (name, estimator)
    transformers: list[tuple[str, Any]] = []

    for child in node.nodes:
        child_steps = list(_iter_steps(child))
        match child_steps:
            case []:
                raise ValueError(f"Can't handle child of Join.\n{child=}\n{node}")
            case [(name, sklearn_thing)]:
                transformers.append((name, sklearn_thing))
            case list():
                sklearn_thing = pipeline_type(child_steps, **pipeline_kwargs)
                transformers.append((child.name, sklearn_thing))

    return (node.name, feature_union_cls(transformers))


def _iter_steps(
    node: Node,
    pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
    **pipeline_kwargs: Any,
) -> Iterator[tuple[str, SklearnItem]]:
    match node:
        case Fixed(item=BaseEstimator()):
            yield (node.name, clone(node.item))
        case Fixed(item=anything):
            yield (node.name, anything)
        case Component():
            yield (node.name, node.build_item())
        case Choice():
            yield from _iter_steps(node.chosen())
        case Sequential():
            for child in node.nodes:
                yield from _iter_steps(child)
        # Bit more involved, we defer to another functino
        case Join():
            yield _process_join(node, pipeline_type=pipeline_type, **pipeline_kwargs)
        case Split():
            yield _process_split(node, pipeline_type=pipeline_type, **pipeline_kwargs)
        case Searchable():
            raise ValueError(f"Can't handle Searchable: {node}")
        case _:
            raise ValueError(f"Can't handle node: {node}")


def build(
    node: Node[Any, Any],
    *,
    pipeline_type: type[SklearnPipelineT] = SklearnPipeline,
    **pipeline_kwargs: Any,
) -> SklearnPipelineT:
    """Build a pipeline into a usable object.

    Args:
        node: The node from which to build a pipeline.
        pipeline_type: The type of pipeline to build. Defaults to the standard
            sklearn pipeline but can be any derivative of that, i.e. ImbLearn's
            pipeline.
        **pipeline_kwargs: The kwargs to pass to the pipeline_type.

    Returns:
        The built pipeline
    """
    return pipeline_type(list(_iter_steps(node)), **pipeline_kwargs)  # type: ignore
