import pytest

from byop.pipeline import choice, step


def test_error_when_uneven_weights() -> None:
    with pytest.raises(ValueError):
        choice("choice", step("a", 1), step("b", 2), weights=[1])


def test_choice_shallow() -> None:
    c = choice(
        "choice",
        step("a", 1),
        step("b", 2) | step("c", 3),
    )

    assert next(c.select({"choice": "a"})) == step("a", 1)
    assert next(c.select({"choice": "b"})) == step("b", 2) | step("c", 3)


def test_choice_deep() -> None:
    c = (
        step("head", "head")
        | choice(
            "choice",
            step("a", 1),
            step("b", 2),
            choice("choice2", step("c", 3), step("d", 4)) | step("e", 5),
        )
        | step("tail", "tail")
    )

    expected = step("head", "head") | step("d", 4) | step("e", 5) | step("tail", "tail")

    chosen = next(c.select({"choice": "choice2", "choice2": "d"}))
    assert chosen == expected
