from __future__ import annotations

import pytest

from amltk import choice, step


def test_error_when_uneven_weights() -> None:
    with pytest.raises(ValueError, match="Weights must be the same length as choices"):
        choice("choice", step("a", object), step("b", object), weights=[1])


def test_choice_shallow() -> None:
    c = choice(
        "choice",
        step("a", object),
        step("b", object) | step("c", object),
    )

    assert next(c.select({"choice": "a"})) == step("a", object)
    assert next(c.select({"choice": "b"})) == step("b", object) | step("c", object)


def test_choice_deep() -> None:
    c = (
        step("head", object)
        | choice(
            "choice",
            step("a", object),
            step("b", object),
            choice("choice2", step("c", object), step("d", object)) | step("e", object),
        )
        | step("tail", object)
    )

    expected = (
        step("head", object)
        | step("d", object)
        | step("e", object)
        | step("tail", object)
    )

    chosen = next(c.select({"choice": "choice2", "choice2": "d"}))
    assert chosen == expected
