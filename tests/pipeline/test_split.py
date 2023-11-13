from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Component, Split


@dataclass
class Thing:
    """A thing."""

    x: int = 1


def test_split_creation_empty() -> None:
    split = Split(name="split")
    assert split.name == "split"
    assert split.nodes == ()


def test_split_construction() -> None:
    split = Split(
        Component(Thing, name="comp1"),
        Component(Thing, name="comp2"),
        name="split",
    )
    assert split.name == "split"
    assert split.nodes == (
        Component(Thing, name="comp1"),
        Component(Thing, name="comp2"),
    )


def test_split_copy() -> None:
    split = Split(Component(Thing, name="comp1", config={"x": 1}), name="split1")
    assert split == split.copy()
