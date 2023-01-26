from itertools import chain, combinations

from pytest_cases import case, parametrize, parametrize_with_cases

from byop import step
from byop.pipeline import Component, Step


@case
@parametrize("size", [1, 3, 10])
def case_component_chain(size: int) -> Component:
    component = Step.join(step(str(i), i) for i in range(size))
    assert isinstance(component, Component)
    return component


@parametrize_with_cases("head", cases=".")
def test_traverse(head: Component) -> None:
    # Component chains with no splits should traverse as they iter
    assert list(head.traverse()) == list(head.iter())


@parametrize_with_cases("head", cases=".")
def test_walk(head: Component) -> None:
    # Components chains with no splits should walk as they iter

    walk = head.walk([], [])

    # Ensure the head has no splits or parents
    splits, parents, the_head = next(walk)
    assert not any(splits)
    assert not any(parents)
    assert the_head == head

    for splits, parents, current_step in walk:
        assert not any(splits)
        assert any(parents)
        # Ensure that the parents are all the steps from the head up to the current step
        assert parents == list(head.head().iter(to=current_step))


@parametrize_with_cases("head", cases=".")
def test_replace_one(head: Component) -> None:
    new_step = step("replacement", "replacement")
    for to_replace in head.iter():
        new_chain = list(head.replace({to_replace.name: new_step}))
        expected = [new_step if s.name == to_replace.name else s for s in head.iter()]
        assert new_chain == expected


@parametrize_with_cases("head", cases=".")
def test_replace_many(head: Component) -> None:
    steps = list(head.iter())
    lens = range(1, len(steps) + 1)
    replacements = [
        {s.name: step(f"{s.name}_r", 0) for s in to_replace}
        for to_replace in chain.from_iterable(
            combinations(steps, length) for length in lens
        )
    ]

    for to_replace in replacements:
        new_chain = list(head.replace(to_replace))
        expected = [to_replace.get(s.name, s) for s in head.iter()]
        assert new_chain == expected


@parametrize_with_cases("head", cases=".")
def test_remove_one(head: Component) -> None:
    for to_remove in head.iter():
        removed_chain = list(head.remove([to_remove.name]))
        expected = [s for s in head.iter() if s.name != to_remove.name]
        assert expected == removed_chain


@parametrize_with_cases("head", cases=".")
def test_remove_many(head: Component) -> None:
    steps = list(head.iter())
    lens = range(1, len(steps) + 1)
    removals = chain.from_iterable(combinations(steps, length) for length in lens)

    for to_remove in removals:
        removed_chain = list(head.remove(list(removals)))
        expected = [s for s in head.iter() if s.name not in to_remove]
        assert expected == removed_chain
