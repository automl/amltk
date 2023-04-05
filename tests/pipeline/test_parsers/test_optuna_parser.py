from __future__ import annotations

import pytest

from byop.parsing import ParseError

try:
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
except ImportError:
    pytest.skip("Optuna not installed", allow_module_level=True)

from byop import Pipeline, choice, split, step


def test_optuna_parser_simple_definitions() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
    )
    seed = 42

    result = pipeline.space(parser="optuna", seed=seed)
    expected = {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
        "b:hp": IntDistribution(1, 10),
        "c:hp": FloatDistribution(1.0, 10.0),
    }
    assert result == expected


def test_optuna_parser_single_hyperparameter_definitions() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
        step("b", 1, space={"hp": IntDistribution(1, 10)}),
        step("c", 1, space={"hp": FloatDistribution(1.0, 10.0)}),
    )

    result = pipeline.space(parser="optuna")
    assert result == {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
        "b:hp": IntDistribution(1, 10),
        "c:hp": FloatDistribution(1.0, 10.0),
    }


def test_optuna_parser_empty_steps() -> None:
    pipeline = Pipeline.create(step("a", 1))

    result = pipeline.space("optuna")
    assert result == {}


def test_optuna_parser_simple_step() -> None:
    pipeline = Pipeline.create(step("a", 1, space={"hp": [1, 2, 3]}))

    result = pipeline.space("optuna")
    assert result == {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
    }


def test_optuna_parser_two_steps() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
        step("b", 2, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
    )

    result = pipeline.space("optuna")
    assert result == {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
        "b:hp": CategoricalDistribution(choices=[1, 2, 3]),
    }


def test_optuna_split() -> None:
    pipeline = Pipeline.create(
        split(
            "split",
            step("a", 1, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
            step("b", 2, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
        )
    )
    result = pipeline.space("optuna")

    assert result == {
        "split:a:hp": CategoricalDistribution(choices=[1, 2, 3]),
        "split:b:hp": CategoricalDistribution(choices=[1, 2, 3]),
    }


def test_optuna_choice_failure() -> None:
    pipeline = Pipeline.create(
        choice(
            "choice",
            step("a", 1, space={"hp1": [1, 2, 3], "hp2": (1, 10)}),
            step("b", 2, space={"hp": [1, 2, 3]}),
        )
    )
    with pytest.raises(
        ParseError, match=r"We currently do not support conditionals with Optuna."
    ):
        pipeline.space("optuna")
