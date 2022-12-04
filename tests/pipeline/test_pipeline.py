import random

import pytest
from more_itertools import first, last
from pytest_cases import case, parametrize, parametrize_with_cases

from byop.pipeline import Pipeline, choice, split, step


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
    pipeline = Pipeline.create(
        head,
        sequential,
        shallow_spread,
        deep_part,
        long_part,
        tail,
    )
    return pipeline


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
def test_indexing(pipeline: Pipeline) -> None:
    for i, s in enumerate(pipeline.steps):
        assert pipeline[i] == s

    assert pipeline[0:3] == pipeline.steps[0:3]
    assert pipeline[::-1] == pipeline.steps[::-1]


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
    selected_step = random.choice(list(pipeline.traverse()))
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


@parametrize_with_cases("pipeline", cases=".", has_tag="shallow")
def test_replace_shallow(pipeline: Pipeline) -> None:
    for s in pipeline.steps:
        new_step = s.mutate(name="new_step")
        new_pipeline = pipeline.replace(s.name, new_step)
        expected_steps = [*s.preceeding(), new_step, *s.proceeding()]
        assert new_pipeline.steps == expected_steps


@pytest.mark.skip("TODO")
@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_replace_deep(pipeline: Pipeline) -> None:
    # TODO
    raise NotImplementedError()


@parametrize_with_cases("pipeline", cases=".", has_tag="shallow")
def test_remove_shallow(pipeline: Pipeline) -> None:
    for s in pipeline.steps:
        new_pipeline = pipeline.remove(s.name)
        expected_steps = [*s.preceeding(), *s.proceeding()]
        assert new_pipeline.steps == expected_steps


@pytest.mark.skip("TODO")
@parametrize_with_cases("pipeline", cases=".", has_tag="deep")
def test_remove_deep(pipeline: Pipeline) -> None:
    # TODO
    raise NotImplementedError()


@parametrize_with_cases("pipeline", cases=".")
def test_validate(pipeline: Pipeline) -> None:
    first_step = pipeline[0]
    new_pipeline = pipeline | step(first_step.name, object())
    with pytest.raises(AssertionError):
        new_pipeline.validate()
