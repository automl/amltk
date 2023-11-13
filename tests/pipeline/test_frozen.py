from __future__ import annotations

from dataclasses import dataclass

from amltk.pipeline import Fixed


@dataclass
class Thing:
    """A thing."""

    x: int = 1


def test_frozen_construction_direct() -> None:
    f = Fixed(Thing(x=1))
    assert f.name == "Thing"
    assert f.item == Thing(x=1)


def test_copy() -> None:
    f = Fixed(Thing(x=1))
    f2 = f.copy()
    assert f == f2
