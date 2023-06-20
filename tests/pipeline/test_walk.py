"""These tests are specifically traversal of steps and pipelines."""
from __future__ import annotations

from pytest_cases import parametrize

from amltk import Pipeline, step


@parametrize("size", [1, 3, 10])
def test_walk_shallow_pipeline(size: int) -> None:
    pipeline = Pipeline.create(step(str(i), i) for i in range(size))

    walk = pipeline.walk()

    # Ensure the head has no splits or parents
    splits, parents, head = next(walk)
    assert not any(splits)
    assert not any(parents)
    assert head == pipeline.head

    for splits, parents, current_step in walk:
        assert not any(splits)
        # Ensure that the parents are all the steps from the head up to the current step
        assert parents == list(pipeline.head.iter(to=current_step))
