from __future__ import annotations

from ConfigSpace import ConfigurationSpace, EqualsCondition

from byop import Pipeline, choice, step
from byop.configspace import ConfigSpaceParser


def test_pipeline_with_2_pipeline_modules() -> None:
    pipeline = Pipeline.create(
        step("1", 1, space={"a": [1, 2, 3]}),
        step("2", 2, space={"b": [4, 5, 6]}),
    )

    module1 = Pipeline.create(
        step("3", 3, space={"c": [7, 8, 9]}),
        step("4", 4, space={"d": [10, 11, 12]}),
        name="module1",
    )

    module2 = Pipeline.create(
        choice(
            "choice",
            step("6", 6, space={"e": [13, 14, 15]}),
            step("7", 7, space={"f": [16, 17, 18]}),
        ),
        name="module2",
    )

    pipeline = pipeline.attach(modules=(module1, module2))
    assert len(pipeline) == 2
    assert len(pipeline.modules) == 2

    space = pipeline.space(parser=ConfigSpaceParser())
    assert isinstance(space, ConfigurationSpace)

    expected_space = {
        "1:a": [1, 2, 3],
        "2:b": [4, 5, 6],
        "module1:3:c": [7, 8, 9],
        "module1:4:d": [10, 11, 12],
        "module2:choice": ["6", "7"],
        "module2:choice:6:e": [13, 14, 15],
        "module2:choice:7:f": [16, 17, 18],
    }
    expected = ConfigurationSpace(expected_space)
    expected.add_conditions(
        [
            EqualsCondition(
                expected["module2:choice:6:e"], expected["module2:choice"], "6"
            ),
            EqualsCondition(
                expected["module2:choice:7:f"], expected["module2:choice"], "7"
            ),
        ]
    )
    assert space == expected
