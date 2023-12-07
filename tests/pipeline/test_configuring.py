from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from amltk.pipeline import Choice, Component, Sequential, Split


def test_heirarchical_str() -> None:
    pipeline = (
        Sequential(name="pipeline")
        >> Component(object, name="one", space={"v": [1, 2, 3]})
        >> Split(
            Component(object, name="x", space={"v": [4, 5, 6]}),
            Component(object, name="y", space={"v": [4, 5, 6]}),
            name="split",
        )
        >> Choice(
            Component(object, name="a", space={"v": [4, 5, 6]}),
            Component(object, name="b", space={"v": [4, 5, 6]}),
            name="choice",
        )
    )
    config = {
        "pipeline:one:v": 1,
        "pipeline:split:x:v": 4,
        "pipeline:split:y:v": 5,
        "pipeline:choice:__choice__": "a",
        "pipeline:choice:a:v": 6,
    }
    result = pipeline.configure(config)

    expected = (
        Sequential(name="pipeline")
        >> Component(object, name="one", config={"v": 1}, space={"v": [1, 2, 3]})
        >> Split(
            Component(object, name="x", config={"v": 4}, space={"v": [4, 5, 6]}),
            Component(object, name="y", config={"v": 5}, space={"v": [4, 5, 6]}),
            name="split",
        )
        >> Choice(
            Component(object, name="a", config={"v": 6}, space={"v": [4, 5, 6]}),
            Component(object, name="b", space={"v": [4, 5, 6]}),
            name="choice",
            config={"__choice__": "a"},
        )
    )

    assert result == expected


def test_heirarchical_str_with_predefined_configs() -> None:
    pipeline = (
        Sequential(name="pipeline")
        >> Component(object, name="one", config={"v": 1})
        >> Split(
            Component(object, name="x"),
            Component(object, name="y", space={"v": [4, 5, 6]}),
            name="split",
        )
        >> Choice(
            Component(object, name="a"),
            Component(object, name="b"),
            name="choice",
        )
    )

    config = {
        "pipeline:one:v": 2,
        "pipeline:one:w": 3,
        "pipeline:split:x:v": 4,
        "pipeline:split:x:w": 42,
        "pipeline:choice:__choice__": "a",
        "pipeline:choice:a:v": 3,
    }

    expected = (
        Sequential(name="pipeline")
        >> Component(object, name="one", config={"v": 2, "w": 3})
        >> Split(
            Component(object, name="x", config={"v": 4, "w": 42}),
            Component(object, name="y", space={"v": [4, 5, 6]}),
            name="split",
        )
        >> Choice(
            Component(object, name="a", config={"v": 3}),
            Component(object, name="b"),
            name="choice",
            config={"__choice__": "a"},
        )
    )

    result = pipeline.configure(config)
    assert result == expected


def test_config_transform() -> None:
    def _transformer_1(_: Mapping, __: Any) -> Mapping:
        return {"hello": "world"}

    def _transformer_2(_: Mapping, __: Any) -> Mapping:
        return {"hi": "mars"}

    pipeline = (
        Sequential(name="pipeline")
        >> Component(
            object,
            name="1",
            space={"a": [1, 2, 3]},
            config_transform=_transformer_1,
        )
        >> Component(
            object,
            name="2",
            space={"b": [1, 2, 3]},
            config_transform=_transformer_2,
        )
    )
    config = {
        "pipeline:1:a": 1,
        "pipeline:2:b": 1,
    }

    expected = (
        Sequential(name="pipeline")
        >> Component(
            object,
            name="1",
            space={"a": [1, 2, 3]},
            config={"hello": "world"},
            config_transform=_transformer_1,
        )
        >> Component(
            object,
            name="2",
            space={"b": [1, 2, 3]},
            config={"hi": "mars"},
            config_transform=_transformer_2,
        )
    )
    assert expected == pipeline.configure(config)


def test_choice_with_config_transform_does_not_get_activated_if_not_chosen() -> None:
    def transform_a(config: Mapping, __: Any) -> Mapping:
        new_config = dict(config)
        # This will cause an error if b is chosen as "a" is not in the config
        new_config["c"] = new_config.pop("a")
        return new_config

    pipeline = Choice(
        Component(
            object,
            name="a",
            config={"z": 5},
            space={"a": [1, 2, 3]},
            config_transform=transform_a,
        ),
        Component(object, name="b", space={"b": [1, 2, 3]}),
        name="choice",
    )
    config_a = {"choice:__choice__": "a", "choice:a:a": 1}
    configured_a = pipeline.configure(config_a)

    expected_a = Choice(
        Component(
            object,
            name="a",
            space={"a": [1, 2, 3]},
            config={"c": 1, "z": 5},
            config_transform=transform_a,
        ),
        Component(object, name="b", space={"b": [1, 2, 3]}),
        name="choice",
        config={"__choice__": "a"},
    )
    assert configured_a == expected_a

    config_b = {"choice:__choice__": "b", "choice:b:b": 1}
    configured_b = pipeline.configure(config_b)
    expected_b = Choice(
        Component(
            object,
            name="a",
            config={"z": 5},
            space={"a": [1, 2, 3]},
            config_transform=transform_a,
        ),
        Component(object, name="b", config={"b": 1}, space={"b": [1, 2, 3]}),
        name="choice",
        config={"__choice__": "b"},
    )
    assert configured_b == expected_b
