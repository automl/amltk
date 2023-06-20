"""The public api for pipeline, steps and components.

Anything changing here is considering a major change
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, TypeVar, overload

from amltk.pipeline.components import Choice, Component, Group, Split
from amltk.pipeline.step import Step

if TYPE_CHECKING:
    from amltk.types import FidT

Space = TypeVar("Space")
T = TypeVar("T")


@overload
def searchable(
    name: str,
    *,
    space: None = None,
    config: Mapping[str, Any] | None = None,
    fidelities: Mapping[str, FidT] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Step[None]:
    ...


@overload
def searchable(
    name: str,
    *,
    space: Space,
    config: Mapping[str, Any] | None = None,
    fidelities: Mapping[str, FidT] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Step[Space]:
    ...


def searchable(
    name: str,
    *,
    space: Space | None = None,
    config: Mapping[str, Any] | None = None,
    fidelities: Mapping[str, FidT] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Step[Space] | Step[None]:
    """A set of searachble items.

    ```python
    from amltk.pipeline import searchable

    s = searchable("parameters", space={"x": (-10.0, 10.0)})
    ```

    Args:
        name: The unique identifier for this set of searachables.
        space: A space asscoiated with this searchable.
        config:
            A config of set values to pass. If any parameter here is also present in
            the space, this will be removed from the space.
        fidelities:
            A fidelity associated with this searchable. This can be a single range
            indicated as a tuple, an ordered list or a mapping from a name to
            any of the above.
        meta: Any metadata to associate with this

    Returns:
        Step
    """
    return Step(
        name=name,
        config=config,
        search_space=space,
        fidelity_space=fidelities,
        meta=meta,
    )


@overload
def step(
    name: str,
    item: T | Callable[..., T],
    *,
    config: Mapping[str, Any] | None = ...,
    fidelities: Mapping[str, FidT] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Component[T, None]:
    ...


@overload
def step(
    name: str,
    item: T | Callable[..., T],
    *,
    space: Space,
    config: Mapping[str, Any] | None = ...,
    fidelities: Mapping[str, FidT] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Component[T, Space]:
    ...


def step(
    name: str,
    item: T | Callable[..., T],
    *,
    space: Space | None = None,
    config: Mapping[str, Any] | None = None,
    fidelities: Mapping[str, FidT] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Component[T, Space] | Component[T, None]:
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
        fidelities:
            A fidelity associated with this searchable. This can be a single range
            indicated as a tuple, an ordered list or a mapping from a name to
            any of the above.
        meta: Any metadata to associate with this

    Returns:
        The component describing this step
    """
    return Component(
        name=name,
        item=item,
        config=config,
        search_space=space,
        fidelity_space=fidelities,
        meta=meta,
    )


def choice(
    name: str,
    *choices: Step,
    weights: Iterable[float] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Choice[None]:
    """Define a choice in a pipeline.

    Args:
        name: The unique name of this step
        *choices: The choices that can be taken
        weights: Weights to assign to each choice
        meta: Any metadata to associate with this

    Returns:
        Choice: Choice component with your choices as possibilities
    """
    weights = list(weights) if weights is not None else None
    if weights and len(weights) != len(choices):
        raise ValueError("Weights must be the same length as choices")

    return Choice(name=name, paths=list(choices), weights=weights, meta=meta)


@overload
def split(
    name: str,
    *paths: Step,
    meta: Mapping[str, Any] | None = ...,
) -> Split[None, None]:
    ...


@overload
def split(
    name: str,
    *paths: Step,
    item: T | Callable[..., T],
    config: Mapping[str, Any] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Split[T, None]:
    ...


@overload
def split(
    name: str,
    *paths: Step,
    item: T | Callable[..., T],
    space: Space,
    config: Mapping[str, Any] | None = ...,
    meta: Mapping[str, Any] | None = ...,
) -> Split[T, Space]:
    ...


def split(
    name: str,
    *paths: Step,
    item: T | Callable[..., T] | None = None,
    space: Space | None = None,
    config: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Split[T, Space] | Split[T, None] | Split[None, None]:
    """Create a Split component, allowing data to flow multiple paths.

    Args:
        name: The unique name of this step
        *paths: The different paths
        item: The item for this step.
        config:
            A config of set values to pass. If any parameter here is also present in
            the space, this will be removed from the space.
        space: A space with which this step can be searched over.
        meta: Any metadata to associate with this

    Returns:
        Split: Split component with your choices as possibilities
    """
    return Split(
        name=name,
        paths=list(paths),
        item=item,
        search_space=space,
        config=config,
        meta=meta,
    )


def group(
    name: str,
    *paths: Step[Space],
    meta: Mapping[str, Any] | None = None,
) -> Group[Space]:
    """Create a Group component, allowing to namespace one or multiple steps.

    Args:
        name: The unique name of this step
        *paths: The different paths
        meta: Any metadata to associate with this

    Returns:
        Group component with your choices as possibilities
    """
    return Group(name=name, paths=list(paths), meta=meta)
