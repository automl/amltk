import pytest

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
        Normal,
    )
except ImportError:
    pytest.skip("ConfigSpace not installed", allow_module_level=True)

from byop import Pipeline, choice, split, step


def test_configspace_parser_simple_definitions() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
    )
    seed = 42

    result = pipeline.space(parser="configspace", seed=seed)
    expected = ConfigurationSpace(
        {
            "a:hp": [1, 2, 3],
            "b:hp": (1, 10),
            "c:hp": (1.0, 10.0),
        },
        seed=seed,
    )
    assert result == expected
    assert result.sample_configuration(5) == expected.sample_configuration(5)


def test_configspace_parser_single_hyperparameter_definitions() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space=Categorical("hp", items=[1, 2, 3])),
        step("b", 1, space=Integer("hp", bounds=(1, 10), log=True)),
        step("c", 1, space=Float("hp", bounds=(1.0, 10.0), distribution=Normal(5, 2))),
    )

    result = pipeline.space(parser="configspace")
    assert result == ConfigurationSpace(
        {
            "a:hp": Categorical("a:hp", items=[1, 2, 3]),
            "b:hp": Integer("b:hp", bounds=(1, 10), log=True),
            "c:hp": Float("c:hp", bounds=(1.0, 10.0), distribution=Normal(5, 2)),
        }
    )


def test_configspace_parser_configspace_empty_steps() -> None:
    pipeline = Pipeline.create(step("a", 1))

    result = pipeline.space("configspace")
    assert result == ConfigurationSpace()


def test_configspace_parser_simple_step() -> None:
    pipeline = Pipeline.create(step("a", 1, space={"hp": [1, 2, 3]}))

    result = pipeline.space("configspace")
    assert result == ConfigurationSpace({"a:hp": [1, 2, 3]})


def test_configspace_parser_two_steps() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space=ConfigurationSpace({"hp": [1, 2, 3]})),
        step("b", 2, space=ConfigurationSpace({"hp": [1, 2, 3]})),
    )

    result = pipeline.space("configspace")
    assert result == ConfigurationSpace({"a:hp": [1, 2, 3], "b:hp": [1, 2, 3]})


def test_configspace_split() -> None:
    pipeline = Pipeline.create(
        split(
            "split",
            step("a", 1, space=ConfigurationSpace({"hp": [1, 2, 3]})),
            step("b", 2, space=ConfigurationSpace({"hp": [1, 2, 3]})),
        )
    )
    result = pipeline.space("configspace")

    assert result == ConfigurationSpace(
        {"split:a:hp": [1, 2, 3], "split:b:hp": [1, 2, 3]}
    )


def test_configspace_choice() -> None:
    pipeline = Pipeline.create(
        choice(
            "choice",
            step("a", 1, space=ConfigurationSpace({"hp1": [1, 2, 3], "hp2": (1, 10)})),
            step("b", 2, space=ConfigurationSpace({"hp": [1, 2, 3]})),
        )
    )
    result = pipeline.space("configspace")

    expected = ConfigurationSpace(
        {
            "choice": ["a", "b"],
            "choice:a:hp1": [1, 2, 3],
            "choice:a:hp2": (1, 10),
            "choice:b:hp": [1, 2, 3],
        }
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice:a:hp1"], expected["choice"], "a"),
            EqualsCondition(expected["choice:a:hp2"], expected["choice"], "a"),
            EqualsCondition(expected["choice:b:hp"], expected["choice"], "b"),
        ]
    )
    assert result == expected


def test_configspace_nested_choice() -> None:
    pipeline = Pipeline.create(
        choice(
            "choice",
            choice(
                "choice2",
                step("a", 3, space=ConfigurationSpace({"hp": [1, 2]})),
                step("b", 2, space=ConfigurationSpace({"hp": [1, 2]})),
            ),
            step("c", 2, space=ConfigurationSpace({"hp": [1, 2]})),
        )
    )
    result = pipeline.space("configspace")

    expected = ConfigurationSpace(
        {
            "choice": ["choice2", "c"],
            "choice:choice2": ["a", "b"],
            "choice:choice2:a:hp": [1, 2],
            "choice:choice2:b:hp": [1, 2],
            "choice:c:hp": [1, 2],
        }
    )
    expected.add_conditions(
        [
            EqualsCondition(expected["choice:choice2"], expected["choice"], "choice2"),
            EqualsCondition(
                expected["choice:choice2:a:hp"], expected["choice:choice2"], "a"
            ),
            EqualsCondition(
                expected["choice:choice2:b:hp"], expected["choice:choice2"], "b"
            ),
            EqualsCondition(expected["choice:c:hp"], expected["choice"], "c"),
        ]
    )
    assert result == expected
