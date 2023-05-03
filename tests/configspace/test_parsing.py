from __future__ import annotations

import pytest

from byop.configspace import ConfigSpaceParser

try:
    from ConfigSpace import ConfigurationSpace, EqualsCondition, ForbiddenEqualsClause
except ImportError:
    pytest.skip("ConfigSpace not installed", allow_module_level=True)

from pytest_cases import case, parametrize_with_cases

from byop.pipeline import Choice, Pipeline, Split, Step, choice, split, step


@case
def case_single_step() -> tuple[Step, ConfigurationSpace]:
    item = step("a", 1, space={"hp": [1, 2, 3]})
    expected = ConfigurationSpace({"a:hp": [1, 2, 3]})
    return item, expected


@case
def case_steps_with_embedded_forbiddens() -> tuple[Step, ConfigurationSpace]:
    space = ConfigurationSpace({"hp": [1, 2, 3], "hp_other": ["a", "b", "c"]})
    space.add_forbidden_clause(ForbiddenEqualsClause(space["hp"], 1))

    item = step("a", 1, space=space)
    expected = ConfigurationSpace({"a:hp": [1, 2, 3]})
    expected.add_forbidden_clause(ForbiddenEqualsClause(expected["a:hp"], 1))

    return item, expected


@case
def case_single_step_two_hp() -> tuple[Step, ConfigurationSpace]:
    item = step("a", 1, space={"hp": [1, 2, 3], "hp2": [1, 2, 3]})
    expected = ConfigurationSpace({"a:hp": [1, 2, 3], "a:hp2": [1, 2, 3]})
    return item, expected


@case
def case_single_step_two_hp_different_types() -> tuple[Step, ConfigurationSpace]:
    item = step("a", 1, space={"hp": [1, 2, 3], "hp2": (1, 10)})
    expected = ConfigurationSpace({"a:hp": [1, 2, 3], "a:hp2": (1, 10)})
    return item, expected


@case
def case_choice() -> tuple[Choice, ConfigurationSpace]:
    item = choice(
        "choice1",
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp2": (1, 10)}),
    )
    expected = ConfigurationSpace(
        {"choice1:a:hp": [1, 2, 3], "choice1:b:hp2": (1, 10), "choice1": ["a", "b"]},
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice1:a:hp"], expected["choice1"], "a"),
            EqualsCondition(expected["choice1:b:hp2"], expected["choice1"], "b"),
        ],
    )
    return item, expected


@case
def case_nested_choices() -> tuple[Choice, ConfigurationSpace]:
    item = choice(
        "choice1",
        choice(
            "choice2",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 1, space={"hp2": (1, 10)}),
        ),
        step("c", 1, space={"hp3": (1, 10)}),
    )
    expected = ConfigurationSpace(
        {
            "choice1:choice2:a:hp": [1, 2, 3],
            "choice1:choice2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1": ["choice2", "c"],
            "choice1:choice2": ["a", "b"],
        },
    )
    expected.add_conditions(
        [
            EqualsCondition(
                expected["choice1:choice2"],
                expected["choice1"],
                "choice2",
            ),
            EqualsCondition(expected["choice1:c:hp3"], expected["choice1"], "c"),
            EqualsCondition(
                expected["choice1:choice2:a:hp"],
                expected["choice1:choice2"],
                "a",
            ),
            EqualsCondition(
                expected["choice1:choice2:b:hp2"],
                expected["choice1:choice2"],
                "b",
            ),
            EqualsCondition(expected["choice1:c:hp3"], expected["choice1"], "c"),
        ],
    )
    return item, expected


@case
def case_nested_choices_with_split() -> tuple[Choice, ConfigurationSpace]:
    item = choice(
        "choice1",
        split(
            "split2",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 1, space={"hp2": (1, 10)}),
        ),
        step("c", 1, space={"hp3": (1, 10)}),
    )
    expected = ConfigurationSpace(
        {
            "choice1:split2:a:hp": [1, 2, 3],
            "choice1:split2:b:hp2": (1, 10),
            "choice1:c:hp3": (1, 10),
            "choice1": ["split2", "c"],
        },
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice1:c:hp3"], expected["choice1"], "c"),
            EqualsCondition(
                expected["choice1:split2:a:hp"],
                expected["choice1"],
                "split2",
            ),
            EqualsCondition(
                expected["choice1:split2:b:hp2"],
                expected["choice1"],
                "split2",
            ),
            EqualsCondition(expected["choice1:c:hp3"], expected["choice1"], "c"),
        ],
    )
    return item, expected


@case
def case_nested_choices_with_split_and_choice() -> tuple[Choice, ConfigurationSpace]:
    item = choice(
        "choice1",
        split(
            "split2",
            choice(
                "choice3",
                step("a", 1, space={"hp": [1, 2, 3]}),
                step("b", 1, space={"hp2": (1, 10)}),
            ),
            step("c", 1, space={"hp3": (1, 10)}),
        ),
        step("d", 1, space={"hp4": (1, 10)}),
    )
    expected = ConfigurationSpace(
        {
            "choice1:split2:choice3:a:hp": [1, 2, 3],
            "choice1:split2:choice3:b:hp2": (1, 10),
            "choice1:split2:c:hp3": (1, 10),
            "choice1:d:hp4": (1, 10),
            "choice1": ["split2", "d"],
            "choice1:split2:choice3": ["a", "b"],
        },
    )

    expected.add_conditions(
        [
            EqualsCondition(expected["choice1:d:hp4"], expected["choice1"], "d"),
            EqualsCondition(
                expected["choice1:split2:choice3"],
                expected["choice1"],
                "split2",
            ),
            EqualsCondition(
                expected["choice1:split2:c:hp3"],
                expected["choice1"],
                "split2",
            ),
            EqualsCondition(
                expected["choice1:split2:choice3:a:hp"],
                expected["choice1:split2:choice3"],
                "a",
            ),
            EqualsCondition(
                expected["choice1:split2:choice3:b:hp2"],
                expected["choice1:split2:choice3"],
                "b",
            ),
            EqualsCondition(expected["choice1:d:hp4"], expected["choice1"], "d"),
        ],
    )

    return item, expected


@case
def case_choice_pipeline() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        choice(
            "choice",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 2, space={"hp": [1, 2, 3]}),
        ),
    )
    expected = ConfigurationSpace(
        {"choice:a:hp": [1, 2, 3], "choice:b:hp": [1, 2, 3], "choice": ["a", "b"]},
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice:a:hp"], expected["choice"], "a"),
            EqualsCondition(expected["choice:b:hp"], expected["choice"], "b"),
        ],
    )
    return pipeline, expected


@case
def case_pipeline_with_choice_modules() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
        modules=[
            choice(
                "choice",
                step("d", 1, space={"hp": (1.0, 10.0)}),
                step("e", 1, space={"hp": (1.0, 10.0)}),
            ),
        ],
    )
    expected = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
            "choice:d:hp": (1.0, 10.0),
            "choice:e:hp": (1.0, 10.0),
            "choice": ["d", "e"],
        },
    )

    expected.add_conditions(
        [
            EqualsCondition(expected["choice:d:hp"], expected["choice"], "d"),
            EqualsCondition(expected["choice:e:hp"], expected["choice"], "e"),
        ],
    )
    return pipeline, expected


@case
def case_joint_steps() -> tuple[Step, ConfigurationSpace]:
    item = step("a", 1, space={"hp": [1, 2, 3]}) | step("b", 1, space={"hp2": (1, 10)})
    expected = ConfigurationSpace({"a:hp": [1, 2, 3], "b:hp2": (1, 10)})
    return item, expected


@case
def case_split_steps() -> tuple[Step, ConfigurationSpace]:
    item = split(
        "split",
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp2": (1, 10)}),
    )
    expected = ConfigurationSpace({"split:a:hp": [1, 2, 3], "split:b:hp2": (1, 10)})
    return item, expected


@case
def case_nested_splits() -> tuple[Split, ConfigurationSpace]:
    item = split(
        "split1",
        split(
            "split2",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 1, space={"hp2": (1, 10)}),
        ),
        step("c", 1, space={"hp3": (1, 10)}),
    )
    expected = ConfigurationSpace(
        {
            "split1:split2:a:hp": [1, 2, 3],
            "split1:split2:b:hp2": (1, 10),
            "split1:c:hp3": (1, 10),
        },
    )
    return item, expected


@case
def case_simple_linear_pipeline() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
    )
    expected = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
        },
    )
    return pipeline, expected


@case
def case_split_pipeline() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        split(
            "split",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 2, space={"hp": [1, 2, 3]}),
        ),
    )
    expected = ConfigurationSpace(
        {
            "split:a:hp": [1, 2, 3],
            "split:b:hp": [1, 2, 3],
        },
    )
    return pipeline, expected


@case
def case_pipeline_with_step_modules() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
        modules=[
            step("d", 1, space={"hp": (1.0, 10.0)}),
            step("e", 1, space={"hp": (1.0, 10.0)}),
        ],
    )
    expected = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
            "d:hp": (1.0, 10.0),
            "e:hp": (1.0, 10.0),
        },
    )
    return pipeline, expected


@case
def case_pipeline_with_pipeline_modules() -> tuple[Pipeline, ConfigurationSpace]:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
        modules=[
            Pipeline.create(
                step("d", 1, space={"hp": (1.0, 10.0)}),
                step("e", 1, space={"hp": (1.0, 10.0)}),
                name="subpipeline",
            ),
        ],
    )
    expected = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
            "subpipeline:d:hp": (1.0, 10.0),
            "subpipeline:e:hp": (1.0, 10.0),
        },
    )
    return pipeline, expected


@parametrize_with_cases("pipeline, expected", cases=".")
def test_parsing_pipeline(pipeline: Pipeline, expected: ConfigurationSpace) -> None:
    parsed_space = pipeline.space(parser=ConfigSpaceParser())
    assert parsed_space == expected


@parametrize_with_cases("pipeline, expected", cases=".")
def test_parsing_pipeline_does_not_mutate_space(
    pipeline: Pipeline,
    expected: ConfigurationSpace,  # noqa: ARG001
) -> None:
    spaces_before = {
        step.qualified_name(): step.search_space for step in pipeline.traverse()
    }
    pipeline.space(parser=ConfigSpaceParser())

    spaces_after = {
        step.qualified_name(): step.search_space for step in pipeline.traverse()
    }
    assert spaces_before == spaces_after


@parametrize_with_cases("pipeline, expected", cases=".")
def test_parsing_twice_produces_same_space(
    pipeline: Pipeline,
    expected: ConfigurationSpace,
) -> None:
    parsed_space = pipeline.space(parser=ConfigSpaceParser())
    parsed_space2 = pipeline.space(parser=ConfigSpaceParser())

    assert parsed_space == expected
    assert parsed_space2 == expected
    assert parsed_space == parsed_space2
