from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case, parametrize_with_cases

from amltk.pipeline import Choice, Component, Fixed, Node, Sequential, Split

try:
    from ConfigSpace import ConfigurationSpace, EqualsCondition, ForbiddenEqualsClause
except ImportError:
    pytest.skip("ConfigSpace not installed", allow_module_level=True)


FLAT = True
NOT_FLAT = False
CONDITIONED = True
NOT_CONDITIONED = False


@dataclass
class Params:
    """A test case for parsing a Node into a ConfigurationSpace."""

    root: Node
    expected: dict[tuple[bool, bool], ConfigurationSpace]


@case
def case_single_frozen() -> Params:
    item = Fixed(object(), name="a")
    space = ConfigurationSpace()
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
    space = ConfigurationSpace({"a:hp": [1, 2, 3]})
    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


@case
def case_steps_with_embedded_forbiddens() -> Params:
    space = ConfigurationSpace({"hp": [1, 2, 3], "hp_other": ["a", "b", "c"]})
    space.add_forbidden_clause(ForbiddenEqualsClause(space["hp"], 2))

    item = Component(object, name="a", space=space)

    with_conditions = ConfigurationSpace(
        {"a:hp": [1, 2, 3], "a:hp_other": ["a", "b", "c"]},
    )
    with_conditions.add_forbidden_clause(
        ForbiddenEqualsClause(with_conditions["a:hp"], 2),
    )

    without_conditions = ConfigurationSpace(
        {"a:hp": [1, 2, 3], "a:hp_other": ["a", "b", "c"]},
    )

    expected = {
        (NOT_FLAT, CONDITIONED): with_conditions,
        (NOT_FLAT, NOT_CONDITIONED): without_conditions,
        (FLAT, CONDITIONED): with_conditions,
        (FLAT, NOT_CONDITIONED): without_conditions,
    }
    return Params(item, expected)  # type: ignore


@case
def case_single_step_two_hp() -> Params:
    item = Component(object, name="a", space={"hp": [1, 2, 3], "hp2": [1, 2, 3]})
    space = ConfigurationSpace({"a:hp": [1, 2, 3], "a:hp2": [1, 2, 3]})

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
    space = ConfigurationSpace({"a:hp": [1, 2, 3], "a:hp2": (1, 10)})
    expected = {
        (NOT_FLAT, CONDITIONED): space,
        (NOT_FLAT, NOT_CONDITIONED): space,
        (FLAT, CONDITIONED): space,
        (FLAT, NOT_CONDITIONED): space,
    }
    return Params(item, expected)  # type: ignore


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
    space = ConfigurationSpace(
        {
            "choice1:a:hp": [1, 2, 3],
            "choice1:b:hp2": (1, 10),
            "choice1:hp3": (1, 10),
            "choice1:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["choice1:a:hp"], space["choice1:__choice__"], "a"),
            EqualsCondition(space["choice1:b:hp2"], space["choice1:__choice__"], "b"),
        ],
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and with conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "choice1:hp3": (1, 10),
            "choice1:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["a:hp"], space["choice1:__choice__"], "a"),
            EqualsCondition(space["b:hp2"], space["choice1:__choice__"], "b"),
        ],
    )
    expected[(FLAT, CONDITIONED)] = space

    # Not Flat and without conditions
    space = ConfigurationSpace(
        {
            "choice1:a:hp": [1, 2, 3],
            "choice1:b:hp2": (1, 10),
            "choice1:hp3": (1, 10),
            "choice1:__choice__": ["a", "b"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "choice1:hp3": (1, 10),
            "choice1:__choice__": ["a", "b"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space
    return Params(item, expected)  # type: ignore


@case
def case_nested_choices() -> Params:
    item = Choice(
        Choice(
            Component(object, name="a", space={"hp": [1, 2, 3]}),
            Component(object, name="b", space={"hp2": (1, 10)}),
            name="choice2",
        ),
        Component(object, name="c", space={"hp3": (1, 10)}),
        name="choice1",
    )

    expected = {}

    # Not flat and with conditions
    space = ConfigurationSpace(
        {
            "choice1:choice2:a:hp": [1, 2, 3],
            "choice1:choice2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1:__choice__": ["c", "choice2"],
            "choice1:choice2:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(
                space["choice1:choice2:__choice__"],
                space["choice1:__choice__"],
                "choice2",
            ),
            EqualsCondition(space["choice1:c:hp3"], space["choice1:__choice__"], "c"),
            EqualsCondition(
                space["choice1:choice2:a:hp"],
                space["choice1:choice2:__choice__"],
                "a",
            ),
            EqualsCondition(
                space["choice1:choice2:b:hp2"],
                space["choice1:choice2:__choice__"],
                "b",
            ),
            EqualsCondition(space["choice1:c:hp3"], space["choice1:__choice__"], "c"),
        ],
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # flat and with conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "choice1:__choice__": ["c", "choice2"],
            "choice2:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(
                space["choice2:__choice__"],
                space["choice1:__choice__"],
                "choice2",
            ),
            EqualsCondition(space["c:hp3"], space["choice1:__choice__"], "c"),
            EqualsCondition(
                space["a:hp"],
                space["choice2:__choice__"],
                "a",
            ),
            EqualsCondition(
                space["b:hp2"],
                space["choice2:__choice__"],
                "b",
            ),
            EqualsCondition(space["c:hp3"], space["choice1:__choice__"], "c"),
        ],
    )
    expected[(FLAT, CONDITIONED)] = space

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "choice1:choice2:a:hp": [1, 2, 3],
            "choice1:choice2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1:__choice__": ["c", "choice2"],
            "choice1:choice2:__choice__": ["a", "b"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "choice1:__choice__": ["c", "choice2"],
            "choice2:__choice__": ["a", "b"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    return Params(item, expected)  # type: ignore


@case
def case_nested_choices_with_split() -> Params:
    item = Choice(
        Split(
            Component(object, name="a", space={"hp": [1, 2, 3]}),
            Component(object, name="b", space={"hp2": (1, 10)}),
            name="split2",
        ),
        Component(object, name="c", space={"hp3": (1, 10)}),
        name="choice1",
    )
    expected = {}

    # Not flat and with conditions
    space = ConfigurationSpace(
        {
            "choice1:split2:a:hp": [1, 2, 3],
            "choice1:split2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1:__choice__": ["c", "split2"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["choice1:c:hp3"], space["choice1:__choice__"], "c"),
            EqualsCondition(
                space["choice1:split2:a:hp"],
                space["choice1:__choice__"],
                "split2",
            ),
            EqualsCondition(
                space["choice1:split2:b:hp2"],
                space["choice1:__choice__"],
                "split2",
            ),
            EqualsCondition(space["choice1:c:hp3"], space["choice1:__choice__"], "c"),
        ],
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # flat and with conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "choice1:__choice__": ["c", "split2"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["c:hp3"], space["choice1:__choice__"], "c"),
            EqualsCondition(space["a:hp"], space["choice1:__choice__"], "split2"),
            EqualsCondition(space["b:hp2"], space["choice1:__choice__"], "split2"),
            EqualsCondition(space["c:hp3"], space["choice1:__choice__"], "c"),
        ],
    )
    expected[(FLAT, CONDITIONED)] = space

    # not flat and without conditions
    space = ConfigurationSpace(
        {
            "choice1:split2:a:hp": [1, 2, 3],
            "choice1:split2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1:__choice__": ["c", "split2"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "choice1:__choice__": ["c", "split2"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    return Params(item, expected)


@case
def case_nested_choices_with_split_and_choice() -> Params:
    item = Choice(
        Split(
            Choice(
                Component(object, name="a", space={"hp": [1, 2, 3]}),
                Component(object, name="b", space={"hp2": (1, 10)}),
                name="choice3",
            ),
            Component(object, name="c", space={"hp3": (1, 10)}),
            name="split2",
        ),
        Component(object, name="d", space={"hp4": (1, 10)}),
        name="choice1",
    )
    expected = {}

    # Not flat and with conditions
    space = ConfigurationSpace(
        {
            "choice1:split2:choice3:a:hp": [1, 2, 3],
            "choice1:split2:choice3:b:hp2": (1, 10),
            "choice1:split2:c:hp3": (1, 10),
            "choice1:d:hp4": (1, 10),
            "choice1:__choice__": ["d", "split2"],
            "choice1:split2:choice3:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["choice1:d:hp4"], space["choice1:__choice__"], "d"),
            EqualsCondition(
                space["choice1:split2:choice3:__choice__"],
                space["choice1:__choice__"],
                "split2",
            ),
            EqualsCondition(
                space["choice1:split2:c:hp3"],
                space["choice1:__choice__"],
                "split2",
            ),
            EqualsCondition(
                space["choice1:split2:choice3:a:hp"],
                space["choice1:split2:choice3:__choice__"],
                "a",
            ),
            EqualsCondition(
                space["choice1:split2:choice3:b:hp2"],
                space["choice1:split2:choice3:__choice__"],
                "b",
            ),
        ],
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and with conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "d:hp4": (1, 10),
            "choice1:__choice__": ["d", "split2"],
            "choice3:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["d:hp4"], space["choice1:__choice__"], "d"),
            EqualsCondition(
                space["choice3:__choice__"],
                space["choice1:__choice__"],
                "split2",
            ),
            EqualsCondition(space["c:hp3"], space["choice1:__choice__"], "split2"),
            EqualsCondition(space["a:hp"], space["choice3:__choice__"], "a"),
            EqualsCondition(space["b:hp2"], space["choice3:__choice__"], "b"),
        ],
    )
    expected[(FLAT, CONDITIONED)] = space

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "choice1:split2:choice3:a:hp": [1, 2, 3],
            "choice1:split2:choice3:b:hp2": (1, 10),
            "choice1:split2:c:hp3": (1, 10),
            "choice1:d:hp4": (1, 10),
            "choice1:__choice__": ["d", "split2"],
            "choice1:split2:choice3:__choice__": ["a", "b"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
            "d:hp4": (1, 10),
            "choice1:__choice__": ["d", "split2"],
            "choice3:__choice__": ["a", "b"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    return Params(item, expected)


@case
def case_sequential_with_choice() -> Params:
    item = Sequential(
        Choice(
            Component(object, name="a", space={"hp": [1, 2, 3]}),
            Component(object, name="b", space={"hp": [1, 2, 3]}),
            name="choice",
        ),
        name="pipeline",
    )
    expected = {}

    # Not flat and with conditions
    space = ConfigurationSpace(
        {
            "pipeline:choice:a:hp": [1, 2, 3],
            "pipeline:choice:b:hp": [1, 2, 3],
            "pipeline:choice:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(
                space["pipeline:choice:a:hp"],
                space["pipeline:choice:__choice__"],
                "a",
            ),
            EqualsCondition(
                space["pipeline:choice:b:hp"],
                space["pipeline:choice:__choice__"],
                "b",
            ),
        ],
    )
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and with conditions
    # Note: For flat configuration, the namespace does not include "pipeline:choice",
    # but just the "choice" to reflect the selected component.
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": [1, 2, 3],
            "choice:__choice__": ["a", "b"],
        },
    )
    space.add_conditions(
        [
            EqualsCondition(space["a:hp"], space["choice:__choice__"], "a"),
            EqualsCondition(space["b:hp"], space["choice:__choice__"], "b"),
        ],
    )
    expected[(FLAT, CONDITIONED)] = space

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "pipeline:choice:a:hp": [1, 2, 3],
            "pipeline:choice:b:hp": [1, 2, 3],
            "pipeline:choice:__choice__": ["a", "b"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": [1, 2, 3],
            "choice:__choice__": ["a", "b"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    return Params(item, expected)


@case
def case_sequential_with_own_search_space() -> Params:
    item = Sequential(
        Component(object, name="a", space={"hp": [1, 2, 3]}),
        Component(object, name="b", space={"hp": (1, 10)}),
        Component(object, name="c", space={"hp": (1.0, 10.0)}),
        name="pipeline",
        space={"something": ["a", "b", "c"]},
    )
    expected = {}

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "pipeline:a:hp": [1, 2, 3],
            "pipeline:b:hp": (1, 10),
            "pipeline:c:hp": (1.0, 10.0),
            "pipeline:something": ["a", "b", "c"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Not flat and with conditions - although it doesn't logically apply here,
    # we still add a dummy condition for the sake of consistency
    space = ConfigurationSpace(
        {
            "pipeline:a:hp": [1, 2, 3],
            "pipeline:b:hp": (1, 10),
            "pipeline:c:hp": (1.0, 10.0),
            "pipeline:something": ["a", "b", "c"],
        },
    )
    space.add_conditions([])
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
            "pipeline:something": ["a", "b", "c"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    # Flat and with conditions - similarly, dummy conditions are added
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
            "pipeline:something": ["a", "b", "c"],
        },
    )
    space.add_conditions([])
    expected[(FLAT, CONDITIONED)] = space

    return Params(item, expected)


@case
def case_nested_splits() -> Params:
    item = Split(
        Split(
            Component(object, name="a", space={"hp": [1, 2, 3]}),
            Component(object, name="b", space={"hp2": (1, 10)}),
            name="split2",
        ),
        Component(object, name="c", space={"hp3": (1, 10)}),
        name="split1",
    )
    expected = {}

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "split1:split2:a:hp": [1, 2, 3],
            "split1:split2:b:hp2": (1, 10),
            "split1:c:hp3": (1, 10),
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Not flat and with conditions - Splits do not have conditions
    space = ConfigurationSpace(
        {
            "split1:split2:a:hp": [1, 2, 3],
            "split1:split2:b:hp2": (1, 10),
            "split1:c:hp3": (1, 10),
        },
    )
    space.add_conditions([])  # No conditions for splits
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    # Flat and with conditions - Conditions would be empty as no __choice__
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp2": (1, 10),
            "c:hp3": (1, 10),
        },
    )
    space.add_conditions([])  # No conditions for splits
    expected[(FLAT, CONDITIONED)] = space

    return Params(item, expected)


@case
def case_sequential_with_split() -> Params:
    pipeline = Sequential(
        Split(
            Component(object, name="a", space={"hp": [1, 2, 3]}),
            Component(object, name="b", space={"hp": [1, 2, 3]}),
            name="split",
        ),
        name="pipeline",
        space={"something": ["a", "b", "c"]},
    )

    expected = {}

    # Not flat and without conditions
    space = ConfigurationSpace(
        {
            "pipeline:split:a:hp": [1, 2, 3],
            "pipeline:split:b:hp": [1, 2, 3],
            "pipeline:something": ["a", "b", "c"],
        },
    )
    expected[(NOT_FLAT, NOT_CONDITIONED)] = space

    # Flat and without conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": [1, 2, 3],
            "pipeline:something": ["a", "b", "c"],
        },
    )
    expected[(FLAT, NOT_CONDITIONED)] = space

    # Not flat and with conditions
    space = ConfigurationSpace(
        {
            "pipeline:split:a:hp": [1, 2, 3],
            "pipeline:split:b:hp": [1, 2, 3],
            "pipeline:something": ["a", "b", "c"],
        },
    )
    space.add_conditions([])  # No conditions to add, but placeholder for consistency
    expected[(NOT_FLAT, CONDITIONED)] = space

    # Flat and with conditions
    space = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": [1, 2, 3],
            "pipeline:something": ["a", "b", "c"],
        },
    )
    space.add_conditions([])  # No conditions to add, but placeholder for consistency
    expected[(FLAT, CONDITIONED)] = space

    return Params(pipeline, expected)


@parametrize_with_cases("test_case", cases=".")
def test_parsing_pipeline(test_case: Params) -> None:
    pipeline = test_case.root

    for (flat, conditioned), expected in test_case.expected.items():
        parsed_space = pipeline.search_space(
            "configspace",
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
            "configspace",
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
            "configspace",
            flat=flat,
            conditionals=conditioned,
        )
        parsed_space2 = pipeline.search_space(
            "configspace",
            flat=flat,
            conditionals=conditioned,
        )
        assert parsed_space == parsed_space2
