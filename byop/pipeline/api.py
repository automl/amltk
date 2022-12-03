"""The public api for pipeline, steps and components

Anything changing here is considering a major change
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar, overload

from byop.pipeline.components import Choice, Component, Searchable, Split
from byop.pipeline.pipeline import Pipeline
from byop.pipeline.step import Key, Step

Space = TypeVar("Space")
T = TypeVar("T")


@overload
def step(
    name: Key, item: T, *, config: Mapping[str, Any] | None = ...
) -> Component[Key, T]:
    ...


@overload
def step(
    name: Key, item: T, *, space: Space, config: Mapping[str, Any] | None = ...
) -> Searchable[Key, T, Space]:
    ...


@overload
def step(
    name: Key,
    item: T,
    *,
    space: Space | None = None,
    config: Mapping[str, Any] | None = ...,
) -> Component[Key, T] | Searchable[Key, T, Space]:
    ...


def step(
    name: Key,
    item: T,
    *,
    space: Space | None = None,
    config: Mapping[str, Any] | None = None,
) -> Component[Key, T] | Searchable[Key, T, Space]:
    """A step in a pipeline.

    Can be joined together with the `|` operator, creating a chain and returning
    a new set of steps, with the first step still at the head.

    ```python
    head = step("1", 1) | step("2", 2) | step("3", 3)
    ```
     Note:
        These are immutable steps, where operations on them will create new instances
        with references to any content they may store. Equality between steps is based
        solely on the name, not their contents or the steps they are linked to.

        For this reason, Pipelines expose a `validate` method that will check that
        the steps in a pipeline are all uniquenly named.

    Args:
        name: The unique identifier for this step.
        item: The item for this step.
        space: A space with which this step can be searched over.
        config:
            A config of set values to pass. If any parameter here is also present in
            the space, this will be removed from the space.

    Returns
    -------
        The component describing this step, a Searchable if it has a space,
        otherwise a Component.
    """
    if space:
        return Searchable(
            name=name,
            item=item,
            config=config,
            space=space,
        )  # type: ignore

    return Component(name=name, item=item, config=config)  # type: ignore


def choice(
    name: Key,
    *choices: Step,
    weights: Iterable[float] | None = None,
) -> Choice:
    """Define a choice in a pipeline

    Args:
        name: The unique name of this step
        *choices: The choices that can be taken
        weights: Weights to assign to each choice, defaults to None

    Returns:
        Choice: Choice component with your choices as possibilities
    """
    weights = list(weights) if weights is not None else None
    if weights and len(weights) != len(choices):
        raise ValueError("Weights must be the same length as choices")

    return Choice(name=name, paths=list(choices), weights=weights)  # type: ignore


def split(
    name: Key,
    *paths: Step,
) -> Split:
    """Create a Split component, allowing data to flow multiple paths.

    Args:
        name (Key): The unique name of this step
        *paths (Step): The different paths

    Returns:
        Split: Split component with your choices as possibilities
    """
    return Split(name=name, paths=list(paths))  # type: ignore


__all__ = ["step", "choice", "split", "Pipeline"]
