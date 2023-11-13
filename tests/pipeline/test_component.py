from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pytest_cases import parametrize

from amltk.pipeline.components import Component


@dataclass
class Thing:
    """A thing."""

    x: int = 1


def thing_maker(x: int = 1) -> Thing:
    return Thing(x)


@parametrize("maker", [Thing, thing_maker])
def test_component_construction(maker: Any) -> None:
    component = Component(maker, name="comp", config={"x": 2})
    assert component.name == "comp"
    assert component.item == maker
    assert component.config == {"x": 2}


@parametrize("maker", [Thing, thing_maker])
def test_component_builds(maker: Callable[[], Thing]) -> None:
    f = Component(maker, name="comp", config={"x": 5})
    obj = f.build_item()
    assert obj == Thing(x=5)


@parametrize("maker", [Thing, thing_maker])
def test_copy(maker: Any) -> None:
    f = Component(maker, name="comp", config={"x": 5}, space={"x": [1, 2, 3]})
    f2 = f.copy()
    assert f == f2
