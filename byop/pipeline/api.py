from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, TypeVar

from byop.pipeline.pipeline import Choice, Component, Configurable, Node, Pipeline

Space = TypeVar("Space")
Item = TypeVar("Item")


def choice(
    name: str,
    *choices: Node,
    weights: Iterable[float] | None = None,
) -> Choice:
    if weights is not None:
        weights = list(weights)
        if len(choices) != len(weights):
            raise ValueError(
                f"Must have weight ({weights}) for each choice ({choices})"
            )

    return Choice(name=name, choices=list(choices), weights=weights)


def step(
    name: str,
    item: Item,
    *,
    space: Space | None = None,
    kwargs: Mapping[str, Any] | None = None,
    inject: Sequence[str] | None = None,
) -> Component[Item] | Configurable[Item, Space]:
    # Done for convenience in Pipeline.build
    if kwargs is None:
        kwargs = {}

    if space is not None:
        return Configurable(
            name=name, item=item, space=space, kwargs=kwargs, inject=inject
        )
    else:
        return Component(name=name, item=item, kwargs=kwargs, inject=inject)


__all__ = ["step", "choice", "Pipeline"]
