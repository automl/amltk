from ConfigSpace import ConfigurationSpace, EqualsCondition

from byop.assembly.space_parsers import ConfigSpaceParser
from byop.pipeline import Pipeline, choice, split, step


def test_configspace_parser_configspace_empty_steps() -> None:
    pipeline = Pipeline.create(step("a", 1))

    result = ConfigSpaceParser.parse(pipeline)
    assert result.unwrap() == ConfigurationSpace()


def test_configspace_parser_simple_step() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space=ConfigurationSpace({"a": [1, 2, 3]})),
    )

    result = ConfigSpaceParser.parse(pipeline)
    assert result.unwrap() == ConfigurationSpace({"a": [1, 2, 3]})


def test_configspace_parser_two_steps() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space=ConfigurationSpace({"a": [1, 2, 3]})),
        step("b", 2, space=ConfigurationSpace({"b": [1, 2, 3]})),
    )

    result = ConfigSpaceParser.parse(pipeline)
    assert result.unwrap() == ConfigurationSpace({"a": [1, 2, 3], "b": [1, 2, 3]})


def test_configspace_split() -> None:
    pipeline = Pipeline.create(
        split(
            "split",
            step("a", 1, space=ConfigurationSpace({"a": [1, 2, 3]})),
            step("b", 2, space=ConfigurationSpace({"b": [1, 2, 3]})),
        )
    )
    result = ConfigSpaceParser.parse(pipeline)

    assert result.unwrap() == ConfigurationSpace(
        {"split:a": [1, 2, 3], "split:b": [1, 2, 3]}
    )


def test_configspace_choice() -> None:
    pipeline = Pipeline.create(
        choice(
            "choice",
            step("a", 1, space=ConfigurationSpace({"a": [1, 2, 3]})),
            step("b", 2, space=ConfigurationSpace({"b": [1, 2, 3]})),
        )
    )
    result = ConfigSpaceParser.parse(pipeline)

    expected = ConfigurationSpace(
        {"choice": ["a", "b"], "choice:a": [1, 2, 3], "choice:b": [1, 2, 3]}
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice:a"], expected["choice"], "a"),
            EqualsCondition(expected["choice:b"], expected["choice"], "b"),
        ]
    )
    assert result.unwrap() == expected


def test_configspace_nested_choice() -> None:
    pipeline = Pipeline.create(
        choice(
            "choice",
            choice(
                "choice2",
                step("a", 3, space=ConfigurationSpace({"a": [1, 2]})),
                step("b", 2, space=ConfigurationSpace({"b": [1, 2]})),
            ),
            step("c", 2, space=ConfigurationSpace({"c": [1, 2]})),
        )
    )
    result = ConfigSpaceParser.parse(pipeline)

    expected = ConfigurationSpace(
        {
            "choice": ["choice2", "c"],
            "choice:choice2": ["a", "b"],
            "choice:choice2:a": [1, 2],
            "choice:choice2:b": [1, 2],
            "choice:c": [1, 2],
        }
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice:choice2"], expected["choice"], "choice2"),
            EqualsCondition(
                expected["choice:choice2:a"], expected["choice:choice2"], "a"
            ),
            EqualsCondition(
                expected["choice:choice2:b"], expected["choice:choice2"], "b"
            ),
            EqualsCondition(expected["choice:c"], expected["choice"], "c"),
        ]
    )
    assert result.unwrap() == expected
