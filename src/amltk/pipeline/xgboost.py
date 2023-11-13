"""Get an XGBoost component for your pipeline.

A [`Component`][amltk.pipeline.Component] wrapped
around [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) with
two possible default spaces `#!python "large"` and `#!python "performant"`
or you own custom `space=`.

See [`amltk.pipeline.xgboost.xgboost_component`][amltk.pipeline.xgboost.xgboost_component]
"""  # noqa: E501


from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal

from xgboost import XGBClassifier, XGBRegressor

from amltk.pipeline import Component


def xgboost_component(
    kind: Literal["classifier", "regressor"],
    name: str | None = None,
    space: Any | Literal["large", "performant"] = "large",
    config: Mapping[str, Any] | None = None,
) -> Component:
    """Create a component with an XGBoost estimator.

    Args:
        kind: The kind of estimator to create, either "classifier" or "regressor".
        name: The name of the step in the pipeline.
        space: The space to use for hyperparameter optimization. Choose from
            "large", "performant" or provide a custom space.

            !!! todo

                Currently `#!python "performant"` links to `#!python "large"` and are
                the same.

            !!! warning "Warning"

                The default space is by no means optimal, please adjust it to your
                needs. You can find the link to the XGBoost parameters here:

                https://xgboost.readthedocs.io/en/stable/parameter.html


        config: The keyword arguments to pass to the XGBoost estimator when it will be
            created. These will be hard set on the estimator and removed from the
            default space if no space is provided.

    Returns:
        A pipeline with an XGBoost estimator.
    """
    estimator_types = {
        "classifier": XGBClassifier,
        "regressor": XGBRegressor,
    }
    config = config or {}
    estimator_type = estimator_types.get(kind)
    if estimator_type is None:
        raise ValueError(
            f"Unknown kind: {kind}, please choose from {list(estimator_types.keys())}",
        )

    if name is None:
        name = str(estimator_type.__name__)

    if isinstance(space, str):
        device = config.get("device", "cpu")
        tree_method = config.get("tree_method", "auto")
        is_classifier = estimator_type is XGBClassifier
        _spaces = {
            "large": xgboost_large_space,
            "performant": xgboost_performant_space,
        }
        space_f = _spaces[space]
        space = space_f(
            is_classifier=is_classifier,
            tree_method=tree_method,
            device=device,
        )
        for key in config:
            space.pop(key, None)
    elif isinstance(space, Mapping):
        overlap = set(space.keys()).intersection(config)
        if any(overlap):
            raise ValueError(
                f"Space and kwargs overlap: {overlap}, please remove one of them",
            )

    return Component(name=name, item=estimator_type, config=config, space=space)


def xgboost_large_space(
    *,
    is_classifier: bool,
    tree_method: str,
    device: str,
) -> dict[str, Any]:
    """Create a large space for XGBoost hyperparameter optimization."""
    # TODO: Do we want a general space kind
    # For now we use ConfigSpace where needed
    from ConfigSpace import Float, Integer

    space = {
        "eta": Float("eta", (1e-3, 1), default=0.3, log=True),
        "min_split_loss": Float("min_split_loss", (0, 20), default=0),
        "max_depth": Integer("max_depth", (1, 20), default=6),
        "min_child_weight": Float("min_child_weight", (0, 20), default=1),
        "max_delta_step": Float("max_delta_step", (0, 10), default=0),
        "subsample": Float("subsample", (1e-5, 1), default=1),
        "colsample_bytree": Float("colsample_bytree", (1e-5, 1), default=1),
        "colsample_bylevel": Float("colsample_bylevel", (1e-5, 1), default=1),
        "colsample_bynode": Float("colsample_bynode", (1e-5, 1), default=1),
        "reg_lambda": Float("reg_lambda", (1e-5, 1e3), default=1),
        "reg_alpha": Float("reg_alpha", (0, 1e3), default=0),
    }

    if tree_method == "hist" and ("cuda" in device or "gpu" in device):
        space["sampling_method"] = ["uniform", "gradient_based"]

    if tree_method in ("hist", "approx"):
        space["max_bin"] = Integer("max_bin", (2, 512), default=256)

    if is_classifier:
        space["scale_pos_weight"] = Float("scale_pos_weight", (1e-5, 1e5), default=1)

    return space


def xgboost_performant_space(
    *,
    is_classifier: bool,
    tree_method: str,
    device: str,
) -> dict[str, Any]:
    """Create a performant space for XGBoost hyperparameter optimization."""
    warnings.warn(
        "This space is not yet optimized for performance"
        " and is subject to change in the future.",
        FutureWarning,
        stacklevel=2,
    )
    return xgboost_large_space(
        is_classifier=is_classifier,
        tree_method=tree_method,
        device=device,
    )
