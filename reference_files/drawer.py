from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import networkx as nx

from byop.pipeline.step import Choice, Component, Searchable, Split, Step

if TYPE_CHECKING:
    import pygraphviz as pgv


def style_conditioned(node: pgv.Node, item: Step) -> None:
    # color is the border color
    node.attr.update({"color": "blue"})


def style_searchable(
    node: pgv.Node,
    item: Searchable,
    formatted: Callable[[Any], str] | None = None,
) -> None:
    label_items: list[str] = [formatted(item.space) if formatted else str(item.space)]

    if any(item.config):
        label_items.append(str(item.config))

    node.attr.update(
        {
            "label": "\n".join(label_items),
            "xlabel": node.name,
            "style": "filled",
            "fillcolor": "gold1",
        }
    )


def style_split(node: pgv.Node, item: Choice) -> None:
    node.attr.update({"style": "filled", "fillcolor": "darkseagreen3"})


def style_choice(node: pgv.Node, item: Choice, A: pgv.AGraph) -> None:
    node.attr.update({"shape": "box", "style": "filled", "fillcolor": "purple"})

    if item.weights:
        frm = node.name
        for choice, weight in zip(item.choices, item.weights):
            to = choice.name
            A.get_edge(frm, to).attr.update(
                {
                    "label": str(weight),
                    "fontcolor": "purple",
                    "fontsize": 8,
                }
            )


def style_component(node: pgv.Node, item: Choice) -> None:
    if not any(item.config):
        return

    node.attr.update({"xlabel": node.name, "label": str(item.config)})


def draw(
    step: Step,
    path: str | Path | None = None,
    *,
    head: bool = True,
    ax: plt.Axes | None = None,
    keep_axis: bool = False,
    layout: str = "dot",
    space_formatter: Callable[Any, str] | None = None,
) -> None:
    _head = step.head if head else step

    G = _head.dag()
    A = nx.nx_agraph.to_agraph(G)

    for node, data in G.nodes(data=True):
        item = data["o"]
        anode = A.get_node(node)
        conditioned_on = data.get("conditioned_on")
        if conditioned_on is not None:
            style_conditioned(anode, item)

        if isinstance(item, Searchable):
            style_searchable(anode, item, space_formatter)
        elif isinstance(item, Choice):
            style_choice(anode, item, A)
        elif isinstance(item, Split):
            style_split(anode, item)
        elif isinstance(item, Component):
            style_component(anode, item)
        else:
            raise ValueError(f"Unknown node ({node}) to style, {item=}")

    A.layout(layout)

    if isinstance(path, str):
        path = Path(path)

    if path and ax:
        A.draw(path)
        with path.open("rb") as f:
            img = plt.imread(f)

        ax.imshow(img)

    elif not path and ax:
        with TemporaryFile() as f:
            A.draw(f, format="jpeg")
            img = plt.imread(f, format="jpeg")

        ax.imshow(img)

    elif path and not ax:
        A.draw(path)

    else:
        raise ValueError("Must provide `path` to save to or an `ax` to plot on")

    if ax and not keep_axis:
        ax.axis("off")
