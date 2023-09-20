from __future__ import annotations

import random

import pytest
from more_itertools import first, last
from pytest_cases import case, parametrize, parametrize_with_cases

from amltk.pipeline import Pipeline, Step, choice, request, split, step


@case(tags="shallow")
@parametrize("size", [1, 3, 10])
def case_shallow_pipeline(size: int) -> Pipeline:
    return Pipeline.create(step(str(i), i) for i in range(size))


@case(tags="deep")
def case_deep_pipeline() -> Pipeline:
    # We want sequential split points
    split1 = split("split1", step("1", 1), step("2", 2))
    split2 = split("split2", step("3", 3), step("4", 4))
    sequential = split1 | split2

    # We want some choices
    choice1 = choice("choice1", step("5", 5), step("6", 6) | step("7", 7))
    choice2 = choice("choice2", step("8", 8), step("9", 9) | step("10", 10))

    # Use these to create at least one layer of depth
    shallow_spread = split("split3", choice1, choice2)

    # Create a very deep part
    deep_part = split("deep1", split("deep2", split("deep3", step("leaf", "leaf"))))

    # Throw on a long part
    long_part = step("l1", 1) | step("l2", 2) | step("l3", 3) | step("l4", 4)
    head = step("head", "head")
    tail = step("tail", "tail")
    return Pipeline.create(
        head,
        sequential,
        shallow_spread,
        deep_part,
        long_part,
        tail,
    )


def test_pipeline_mixture_of_steps() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    s3 = step("3", 3)
    s4 = step("4", 4)

    pipeline = Pipeline.create(s1, s2 | s3, s4)

    assert list(pipeline) == [s1, s2, s3, s4]


@parametrize_with_cases("pipeline", cases=".")
def test_pipeline_has_steps(pipeline: Pipeline) -> None:
    assert list(pipeline) == list(pipeline.steps)


@parametrize_with_cases("pipeline", cases=".")
def test_head(pipeline: Pipeline) -> None:
    assert pipeline.head == pipeline.steps[0] == first(pipeline)


@parametrize_with_cases("pipeline", cases=".")
def test_tail(pipeline: Pipeline) -> None:
    assert pipeline.tail == pipeline.steps[-1] == last(pipeline)


@parametrize_with_cases("pipeline", cases=".")
def test_len(pipeline: Pipeline) -> None:
    assert len(pipeline) == len(pipeline.steps)


@parametrize_with_cases("pipeline", cases=".")
def test_iter_shallow(pipeline: Pipeline) -> None:
    assert all(a == b for a, b in zip(pipeline, pipeline.steps))


@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_traverse_contains_deeper_items_than_iter(pipeline: Pipeline) -> None:
    # TODO: This should probably be tested better and with a specific example
    shallow_items = set(pipeline.iter())
    deep_items = set(pipeline.traverse())

    assert deep_items.issuperset(shallow_items)
    assert len(deep_items) > len(shallow_items)


@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_traverse_contains_no_duplicates(pipeline: Pipeline) -> None:
    seen: set[str] = set()
    for item in pipeline.traverse():
        assert item.name not in seen
        seen.add(item.name)


@parametrize_with_cases("pipeline", cases=".")
def test_find_shallow_success(pipeline: Pipeline) -> None:
    for selected_step in pipeline.steps:
        assert pipeline.find(selected_step.name) == selected_step


@parametrize_with_cases("pipeline", cases=".")
def test_find_default(pipeline: Pipeline) -> None:
    o = object()
    assert pipeline.find("dummy", default=o) is o


@parametrize_with_cases("pipeline", cases=".")
def test_find_not_present(pipeline: Pipeline) -> None:
    assert pipeline.find("dummy") is None


@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_find_deep(pipeline: Pipeline) -> None:
    selected_step = random.choice(list(pipeline.traverse()))  # noqa: S311
    assert pipeline.find(selected_step.name) == selected_step


def test_or_operator() -> None:
    p1 = Pipeline.create(step("1", 1) | step("2", 2))
    p2 = Pipeline.create(step("3", 3) | step("4", 4))
    s = step("hello", "world")
    pnew = p1 | p2 | s
    assert pnew == p1 | p2 | s


def test_append() -> None:
    p1 = Pipeline.create(step("1", 1) | step("2", 2))
    p2 = Pipeline.create(step("3", 3) | step("4", 4))
    s = step("hello", "world")
    pnew = p1.append(p2).append(s)
    # Need to make sure they have the same name for pipeline equality
    assert pnew == Pipeline.create(p1, p2, s, name=pnew.name)


@parametrize_with_cases("pipeline", cases=".")
def test_replace(pipeline: Pipeline) -> None:
    new_step = step("replacement", "replacement")
    for selected_step in pipeline.traverse():
        assert selected_step in pipeline
        assert new_step not in pipeline

        new_pipeline = pipeline.replace(selected_step.name, new_step)
        assert selected_step not in new_pipeline
        assert new_step in new_pipeline

        replacement_step = new_pipeline.find(new_step.name)
        assert replacement_step is not None
        assert replacement_step == new_step
        assert replacement_step.nxt == selected_step.nxt
        assert replacement_step.prv == selected_step.prv


@parametrize_with_cases("pipeline", cases=".", has_tag="shallow")
def test_remove_shallow(pipeline: Pipeline) -> None:
    for s in pipeline.steps:
        new_pipeline = pipeline.remove(s.name)
        expected_steps = [*s.preceeding(), *s.proceeding()]
        assert new_pipeline.steps == expected_steps


@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_remove_deep(pipeline: Pipeline) -> None:
    for selected_step in pipeline.traverse():
        selected_prv = selected_step.prv
        selected_nxt = selected_step.nxt

        assert selected_step in pipeline
        new_pipeline = pipeline.remove(selected_step.name)

        assert selected_step not in new_pipeline

        # Ensure that the previous and next steps are still connected
        if selected_prv is not None:
            new_prv = new_pipeline.find(selected_prv.name)
            assert new_prv is not None
            assert new_prv.nxt == selected_nxt

        if selected_nxt is not None:
            new_nxt = new_pipeline.find(selected_nxt.name)
            assert new_nxt is not None
            assert new_nxt.prv == selected_prv


@parametrize_with_cases("pipeline", cases=".")
def test_duplicate_name_error(pipeline: Pipeline) -> None:
    first_step = pipeline.head
    name = first_step.name
    with pytest.raises(Step.DuplicateNameError):
        pipeline | step(name, object())  # pyright: reportUnusedExpression=false


def test_qualified_name() -> None:
    pipeline = Pipeline.create(
        step("1", 1)
        | step("2", 2)
        | split(
            "split",
            step("split1", "split1") | step("split2", "split2"),
        )
        | choice(
            "3",
            step("4", 4),
            step("5", 5),
        ),
    )
    assert pipeline.qualified_name("1") == "1"
    assert pipeline.qualified_name("2") == "2"
    assert pipeline.qualified_name("split") == "split"
    assert pipeline.qualified_name("split1") == "split:split1"
    assert pipeline.qualified_name("split2") == "split:split2"
    assert pipeline.qualified_name("3") == "3"
    assert pipeline.qualified_name("4") == "3:4"
    assert pipeline.qualified_name("5") == "3:5"


@parametrize_with_cases("pipeline", cases=".")
def test_renaming_function(pipeline: Pipeline) -> None:
    new_name = "replaced_name"
    x = step("nothing", "nothing")

    assert pipeline.replace(x.name, x, name=new_name).name == new_name
    assert pipeline.remove(x.name, name=new_name).name == new_name
    assert pipeline.append(x, name=new_name).name == new_name
    assert pipeline.select({x.name: x.name}, name=new_name).name == new_name


def test_param_requests() -> None:
    pipeline = Pipeline.create(
        step("1", 1, config={"seed": request("seed1")})
        | step("2", 2, config={"seed": request("seed2")})
        | split(
            "split",
            (
                step(
                    "split1",
                    42,
                    config={
                        "seed": request("seed1", required=True),
                    },
                )
                | step(
                    "split2",
                    45,
                    config={"seed": request("seed4", default=4)},
                )
            ),
        )
        | choice(
            "3",
            step("4", 4, config={"seed": None}),
            step("5", 5),
        ),
    )
    configured_pipeline = pipeline.configure(config={}, params={"seed1": 1, "seed2": 2})

    assert configured_pipeline.config() == {
        "1:seed": 1,
        "2:seed": 2,
        "split:split1:seed": 1,
        "split:split2:seed": 4,
        "3:4:seed": None,
    }

    assert configured_pipeline == Pipeline.create(
        step("1", 1, config={"seed": 1})
        | step("2", 2, config={"seed": 2})
        | split(
            "split",
            (
                step("split1", 42, config={"seed": 1})
                | step("split2", 45, config={"seed": 4})
            ),
        )
        | choice(
            "3",
            step("4", 4, config={"seed": None}),
            step("5", 5),
        ),
        name=pipeline.name,
    )
