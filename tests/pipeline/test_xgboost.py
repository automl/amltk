from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pytest
from pytest_cases import parametrize

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    pytest.skip("xgboost not installed", allow_module_level=True)

from amltk.pipeline.xgboost import xgboost_component


@parametrize(
    "kind, expected_type",
    [("classifier", XGBClassifier), ("regressor", XGBRegressor)],
)
@parametrize("space", ["large", "performant"])
def test_xgboost_pipeline_default_creation(
    kind: Literal["classifier", "regressor"],
    expected_type: type,
    space: str,
) -> None:
    name = f"name_{kind}_{space}"
    xgb = xgboost_component(name=name, kind=kind, space=space)
    assert xgb.item is expected_type
    assert isinstance(xgb.space, Mapping)
    assert len(xgb.space) > 1
    assert xgb.name == name

    model = xgb.build_item()
    assert isinstance(model, expected_type)


def test_xgboost_custom_config() -> None:
    eta = 0.341
    xgb = xgboost_component("classifier", config={"eta": eta})

    assert isinstance(xgb.space, Mapping)
    assert "eta" not in xgb.space

    assert xgb.config is not None
    assert xgb.config["eta"] == eta


def test_xgboost_overlapping_config_and_space_raises() -> None:
    with pytest.raises(ValueError, match="Space and kwargs overlap:"):
        xgboost_component(
            "classifier",
            space={"eta": (0.0, 1.0)},
            config={"eta": 0.341},
        )
