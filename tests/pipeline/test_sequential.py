from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Component, Sequential


@dataclass
class Thing:
    """A thing."""

    x: int = 1


def test_sequential_construction_empty() -> None:
    seq = Sequential(name="seq")
    assert seq.name == "seq"
    assert seq.nodes == ()


def test_sequential_construction() -> None:
    seq = Sequential(
        Component(Thing, name="comp1"),
        Component(Thing, name="comp2"),
        name="seq",
    )
    assert seq.name == "seq"
    assert seq.nodes == (Component(Thing, name="comp1"), Component(Thing, name="comp2"))


def test_sequential_copy() -> None:
    seq1 = Sequential(Component(Thing, name="comp1", config={"x": 1}), name="seq1")
    seq2 = Sequential(Component(Thing, name="comp2", config={"x": 1}), name="seq2")

    assert seq1 == seq1.copy()
    assert seq2 == seq2.copy()

    seq3 = seq1 >> seq2
    assert seq3 == seq3.copy()


def test_sequential_rshift() -> None:
    """__rshift__ changes behavior when compared to other nodes."""
    seq = (
        Sequential(name="seq")
        >> Component(Thing, name="comp1", config={"x": 1})
        >> Component(Thing, name="comp2", config={"x": 1})
        >> Component(Thing, name="comp3", config={"x": 1})
    )

    assert seq == Sequential(
        Component(Thing, name="comp1", config={"x": 1}),
        Component(Thing, name="comp2", config={"x": 1}),
        Component(Thing, name="comp3", config={"x": 1}),
        name="seq",
    )
