import pytest

from byop.pipeline import choice, step


def test_error_when_uneven_weights() -> None:
    with pytest.raises(ValueError):
        choice("choice", step("a", 1), step("b", 2), weights=[1])
