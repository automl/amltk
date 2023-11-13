from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Choice, Component


@dataclass
class Thing1:
    """A thing."""

    x: int = 1


@dataclass
class Thing2:
    """A thing."""

    x: int = 2


def test_choice_creation_empty() -> None:
    choice = Choice()
    assert choice.nodes == ()


def test_choice_construction() -> None:
    choice = Choice(Thing1, Thing2)
    assert choice.nodes == (Component(Thing1), Component(Thing2))


def test_choice_copy() -> None:
    choice = Choice(Component(Thing2, config={"x": 1}))
    assert choice == choice.copy()


def test_choice_or() -> None:
    """__or__ changes behavior when compared to other nodes."""
    choice = (
        Choice(name="choice")
        | Component(Thing1, name="comp1", config={"x": 1})
        | Component(Thing1, name="comp2", config={"x": 1})
        | Component(Thing1, name="comp3", config={"x": 1})
    )

    assert choice == Choice(
        Component(Thing1, name="comp1", config={"x": 1}),
        Component(Thing1, name="comp2", config={"x": 1}),
        Component(Thing1, name="comp3", config={"x": 1}),
        name="choice",
    )


def test_choice_configured_gives_chosen_node() -> None:
    choice = (
        Choice(name="choice_thing")
        | Component(Thing1, name="comp1", config={"x": 1})
        | Component(Thing1, name="comp2", config={"x": 1})
        | Component(Thing1, name="comp3", config={"x": 1})
    )
    configured_choice = choice.configure({"__choice__": "comp2"})

    assert configured_choice == Choice(
        Component(Thing1, name="comp1", config={"x": 1}),
        Component(Thing1, name="comp2", config={"x": 1}),
        Component(Thing1, name="comp3", config={"x": 1}),
        name="choice_thing",
        config={"__choice__": "comp2"},
    )

    assert configured_choice.chosen() == Component(
        Thing1,
        name="comp2",
        config={"x": 1},
    )
