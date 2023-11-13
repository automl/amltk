from __future__ import annotations

from amltk.pipeline import Searchable


def test_searchable_construction() -> None:
    component = Searchable({"x": ["red", "green", "blue"]}, name="searchable")
    assert component.name == "searchable"
    assert component.space == {"x": ["red", "green", "blue"]}


def test_searchable_copyable() -> None:
    component = Searchable({"x": ["red", "green", "blue"]}, name="searchable")
    assert component.copy() == component
