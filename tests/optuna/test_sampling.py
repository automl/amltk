from __future__ import annotations

from pytest_cases import case, parametrize, parametrize_with_cases

from byop.optuna.space import OptunaSpaceAdapter
from byop.pipeline import Pipeline, Split, Step, split, step


@case
def case_single_step() -> Step:
    return step("a", 1, space={"hp": [1, 2, 3]})


@case
def case_single_step_two_hp() -> Step:
    return step("a", 1, space={"hp": [1, 2, 3], "hp2": [1, 2, 3]})


@case
def case_single_step_two_hp_different_types() -> Step:
    return step("a", 1, space={"hp": [1, 2, 3], "hp2": (1, 10)})


@case
def case_joint_steps() -> Step:
    return step("a", 1, space={"hp": [1, 2, 3]}) | step("b", 1, space={"hp2": (1, 10)})


@case
def case_split_steps() -> Step:
    return split(
        "split",
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp2": (1, 10)}),
    )


@case
def case_nested_splits() -> Split:
    return split(
        "split1",
        split(
            "split2",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 1, space={"hp2": (1, 10)}),
        ),
        step("c", 1, space={"hp3": (1, 10)}),
    )


@case
def case_simple_linear_pipeline() -> Pipeline:
    return Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
    )


@case
def case_split_pipeline() -> Pipeline:
    return Pipeline.create(
        split(
            "split",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 2, space={"hp": [1, 2, 3]}),
        ),
    )


@case
def case_pipeline_with_step_modules() -> Pipeline:
    return Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
        modules=[
            step("d", 1, space={"hp": (1.0, 10.0)}),
            step("e", 1, space={"hp": (1.0, 10.0)}),
        ],
    )


@case
def case_pipeline_with_pipeline_modules() -> Pipeline:
    return Pipeline.create(
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 1, space={"hp": (1, 10)}),
        step("c", 1, space={"hp": (1.0, 10.0)}),
        modules=[
            Pipeline.create(
                step("d", 1, space={"hp": (1.0, 10.0)}),
                step("e", 1, space={"hp": (1.0, 10.0)}),
            ),
        ],
    )


@parametrize("n", [None, 5, 10])
@parametrize_with_cases("item", cases=".", prefix="case_")
def test_sample_with_seed_returns_same_results(
    item: Pipeline | Step,
    n: int | None,
) -> None:
    space = item.space(parser=OptunaSpaceAdapter())

    configs_1 = item.sample(space, sampler=OptunaSpaceAdapter(), seed=1, n=n)
    configs_2 = item.sample(space, sampler=OptunaSpaceAdapter(), seed=1, n=n)

    assert configs_1 == configs_2
