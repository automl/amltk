from __future__ import annotations

import pytest

from byop.optuna import OptunaParser
from byop.pipeline import Parser

try:
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
except ImportError:
    pytest.skip("Optuna not installed", allow_module_level=True)

from byop.pipeline import Pipeline, choice, split, step


def test_optuna_parser_simple_definitions() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
    )
    result = pipeline.space(parser=OptunaParser())
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

    result = pipeline.space(parser=OptunaParser())
    assert result == {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
        "b:hp": IntDistribution(1, 10),
        "c:hp": FloatDistribution(1.0, 10.0),
    }


def test_optuna_parser_empty_steps() -> None:
    pipeline = Pipeline.create(step("a", 1))

    result = pipeline.space(OptunaParser())
    assert result == {}


def test_optuna_parser_simple_step() -> None:
    pipeline = Pipeline.create(step("a", 1, space={"hp": [1, 2, 3]}))

    result = pipeline.space(OptunaParser())
    assert result == {
        "a:hp": CategoricalDistribution(choices=[1, 2, 3]),
    }


def test_optuna_parser_two_steps() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
        step("b", 2, space={"hp": CategoricalDistribution(choices=[1, 2, 3])}),
    )

    result = pipeline.space(OptunaParser())
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
    result = pipeline.space(OptunaParser())

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
    with pytest.raises(Parser.Error):
        pipeline.space(OptunaParser())
