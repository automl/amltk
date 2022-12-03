from pytest_cases import case, parametrize, parametrize_with_cases

from byop.pipeline import step
from byop.pipeline.components import Component, Step


@case
@parametrize("size", [1, 3, 10])
def case_component_chain(size: int) -> Component:
    component = Step.join(step(str(i), i) for i in range(size))
    assert isinstance(component, Component)
    return component


@parametrize_with_cases("component", cases=".")
def test_traverse(component: Component) -> None:
    # Component chains with no splits should traverse as they iter
    assert list(component.traverse()) == list(component.iter())


@parametrize_with_cases("component", cases=".")
def test_walk(component: Component) -> None:
    # Components chains with no splits should walk as they iter

    walk = component.walk()

    # Ensure the head has no splits or parents
    splits, parents, head = next(walk)
    assert splits is None
    assert parents is None
    assert head == component

    for splits, parents, current_step in walk:
        assert splits is None
        assert parents is not None
        # Ensure that the parents are all the steps from the head up to the current step
        assert list(parents) == list(component.head().iter(to=current_step))
