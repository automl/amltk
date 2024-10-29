# TODO: Fill this in
from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case, parametrize_with_cases

from amltk.pipeline import Component, Fixed, Node
from amltk.pipeline.components import Choice

try:
    from optuna.distributions import CategoricalDistribution, IntDistribution

    from amltk.pipeline.parsers.optuna import OptunaSearchSpace
except ImportError:
    pytest.skip("Optuna not installed", allow_module_level=True)


FLAT = True
NOT_FLAT = False
CONDITIONED = True
NOT_CONDITIONED = False


@dataclass
class Params:
    """A test case for parsing a Node into a ConfigurationSpace."""

    root: Node
    expected: dict[tuple[bool, bool], OptunaSearchSpace]


@case
def case_single_frozen() -> Params:
    item = Fixed(object(), name="a")
    space = OptunaSearchSpace()
    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


@case
def case_single_component() -> Params:
    item = Component(object, name="a", space={"hp": [1, 2, 3]})
    space = OptunaSearchSpace({"a:hp": CategoricalDistribution([1, 2, 3])})
    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


@case
def case_single_step_two_hp() -> Params:
    item = Component(object, name="a", space={"hp": [1, 2, 3], "hp2": [1, 2, 3]})
    space = OptunaSearchSpace(
        {
            "a:hp": CategoricalDistribution([1, 2, 3]),
            "a:hp2": CategoricalDistribution([1, 2, 3]),
        },
    )

    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


@case
def case_single_step_two_hp_different_types() -> Params:
    item = Component(object, name="a", space={"hp": [1, 2, 3], "hp2": (1, 10)})
    space = OptunaSearchSpace(
        {"a:hp": CategoricalDistribution([1, 2, 3]), "a:hp2": IntDistribution(1, 10)},
    )
    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


# TODO: Testing for with and without conditions does not really make sense here
@case
def case_choice() -> Params:
    item = Choice(
        Component(object, name="a", space={"hp": [1, 2, 3]}),
        Component(object, name="b", space={"hp2": (1, 10)}),
        name="choice1",
        space={"hp3": (1, 10)},
    )

    expected = {}

    # Not flat and with conditions
    space = OptunaSearchSpace(
        {
            "choice1:a:hp": CategoricalDistribution([1, 2, 3]),
            "choice1:b:hp2": IntDistribution(1, 10),
            "choice1:hp3": IntDistribution(1, 10),
            "choice1:__choice__": CategoricalDistribution(["a", "b"]),
        },
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and with conditions
    space = OptunaSearchSpace(
        {
            "a:hp": CategoricalDistribution([1, 2, 3]),
            "b:hp2": IntDistribution(1, 10),
            "choice1:hp3": IntDistribution(1, 10),
            "choice1:__choice__": CategoricalDistribution(["a", "b"]),
        },
    )
    expected[(FLAT, CONDITIONED)] = space

    # Not Flat and without conditions
    space = OptunaSearchSpace(
        {
            "choice1:a:hp": CategoricalDistribution([1, 2, 3]),
            "choice1:b:hp2": IntDistribution(1, 10),
            "choice1:hp3": IntDistribution(1, 10),
            "choice1:__choice__": CategoricalDistribution(["a", "b"]),
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Flat and without conditions
    space = OptunaSearchSpace(
        {
            "a:hp": CategoricalDistribution([1, 2, 3]),
            "b:hp2": IntDistribution(1, 10),
            "choice1:hp3": IntDistribution(1, 10),
            "choice1:__choice__": CategoricalDistribution(["a", "b"]),
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space
    return Params(item, expected)  # type: ignore


@parametrize_with_cases("test_case", cases=".")
def test_parsing_pipeline(test_case: Params) -> None:
    pipeline = test_case.root

    for (flat, conditioned), expected in test_case.expected.items():
        parsed_space = pipeline.search_space(
            "optuna",
            flat=flat,
            conditionals=conditioned,
        )
        assert (
            parsed_space == expected
        ), f"Failed for {flat=}, {conditioned=}.\n{parsed_space}\n{expected}"


@parametrize_with_cases("test_case", cases=".")
def test_parsing_does_not_mutate_space_of_nodes(test_case: Params) -> None:
    pipeline = test_case.root
    spaces_before = {tuple(path): step.space for path, step in pipeline.walk()}

    for (flat, conditioned), _ in test_case.expected.items():
        pipeline.search_space(
            "optuna",
            flat=flat,
            conditionals=conditioned,
        )
        spaces_after = {tuple(path): step.space for path, step in pipeline.walk()}
        assert spaces_before == spaces_after


@parametrize_with_cases("test_case", cases=".")
def test_parsing_twice_produces_same_space(test_case: Params) -> None:
    pipeline = test_case.root

    for (flat, conditioned), _ in test_case.expected.items():
        parsed_space = pipeline.search_space(
            "optuna",
            flat=flat,
            conditionals=conditioned,
        )
        parsed_space2 = pipeline.search_space(
            "optuna",
            flat=flat,
            conditionals=conditioned,
        )
        assert parsed_space == parsed_space2
