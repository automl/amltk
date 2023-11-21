from __future__ import annotations

from dataclasses import dataclass

import pytest

from amltk.exceptions import RequestNotMetError
from amltk.pipeline import Component, request


@dataclass
class RandomModel:
    """A model that makes random predictions."""

    random_state: int | None = None


def test_request_correctly_sets_config_on_object() -> None:
    model = Component(RandomModel, config={"random_state": request("seed")})

    expected_model = Component(RandomModel, config={"random_state": 42})
    assert model.configure({}, params={"seed": 42}) == expected_model


def test_request_with_default_sets_default() -> None:
    model = Component(
        RandomModel,
        config={"random_state": request("seed", default=1337)},
    )

    expected_model = Component(RandomModel, config={"random_state": 1337})
    assert model.configure({}) == expected_model


def test_request_without_default_raises_when_not_provided() -> None:
    model = Component(
        RandomModel,
        config={"random_state": request("seed")},
    )

    with pytest.raises(RequestNotMetError):
        model.configure({})
