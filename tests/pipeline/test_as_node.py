from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Choice, Component, Fixed, Join, Sequential, as_node


def test_as_node_returns_copy_of_node() -> None:
    c = Component(int)

    out = as_node(c)
    assert c == out

    # Should be a copied object
    assert id(out) != id(c)


def test_as_node_with_tuple_returns_join() -> None:
    c1 = Component(int)
    c2 = Component(str)
    c3 = Component(bool)

    out = as_node((c1, c2, c3))
    expected = Join(c1, c2, c3)
    assert out.nodes == expected.nodes


def test_as_node_with_set_returns_choice() -> None:
    c1 = Component(int)
    c2 = Component(str)
    c3 = Component(bool)

    # A choice will sort the nodes such that their
    # order is always consistent, even though we provided
    # a set.
    out = as_node({c1, c2, c3})
    expected = Choice(c1, c2, c3)

    assert out.nodes == expected.nodes


def test_as_node_with_list_returns_sequential() -> None:
    c1 = Component(int)
    c2 = Component(str)
    c3 = Component(bool)

    out = as_node([c1, c2, c3])
    expected = Sequential(c1, c2, c3)
    assert out.nodes == expected.nodes


@dataclass
class MyThing:
    """Docstring."""

    x: int = 1

    def __call__(self) -> None:
        """Dummy to try trick frozen into thinking it's a function, it should not."""


def test_as_node_with_constructed_object_returns_frozen() -> None:
    thing = MyThing(1)
    out = as_node(thing)
    assert out == Fixed(thing)


def create_a_thing() -> MyThing:
    return MyThing(1)


def test_as_node_with_callable_function_returns_component() -> None:
    out = as_node(create_a_thing)
    assert out == Component(create_a_thing)


def test_as_node_with_class_returns_component() -> None:
    out = as_node(MyThing)
    assert out == Component(MyThing)
