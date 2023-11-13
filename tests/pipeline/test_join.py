from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Component, Join


@dataclass
class Thing:
    """A thing."""

    x: int = 1


def test_join_creation_empty() -> None:
    j = Join()
    assert j.nodes == ()


def test_join_construction() -> None:
    j = Join(
        Component(Thing, name="comp1"),
        Component(Thing, name="comp2"),
        name="join",
    )
    assert j.name == "join"
    assert j.nodes == (Component(Thing, name="comp1"), Component(Thing, name="comp2"))


def test_join_copy() -> None:
    join1 = Join(Component(Thing, name="comp1", config={"x": 1}), name="join1")
    join2 = Join(Component(Thing, name="comp2", config={"x": 1}), name="join2")

    assert join1 == join1.copy()
    assert join2 == join2.copy()

    join3 = join1 & join2
    assert join3 == join3.copy()


def test_join_and() -> None:
    """__and__ changes behavior when compared to other nodes."""
    join = (
        Join(name="join")
        & Component(Thing, name="comp1", config={"x": 1})
        & Component(Thing, name="comp2", config={"x": 1})
        & Component(Thing, name="comp3", config={"x": 1})
    )

    assert join == Join(
        Component(Thing, name="comp1", config={"x": 1}),
        Component(Thing, name="comp2", config={"x": 1}),
        Component(Thing, name="comp3", config={"x": 1}),
        name="join",
    )
