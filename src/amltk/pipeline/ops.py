"""Operations on pipelines."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from itertools import product
from typing import TypeVar

from amltk.pipeline.components import Choice
from amltk.pipeline.node import Node

NodeT1 = TypeVar("NodeT1", bound=Node)
NodeT2 = TypeVar("NodeT2", bound=Node)


def factorize(
    node: NodeT1,
    *,
    factor_by: Callable[[Node], bool] | None = None,
    assign_child: Callable[[NodeT2, Node], NodeT2] | None = None,
) -> Iterator[NodeT1]:
    """Factorize a pipeline into all possibilities of its children.

    When dealing with a large pipeline with many choices at various levels,
    it can be useful to factorize the pipeline into all possible pipelines.
    This effectively returns a new pipeline for every possible choice in the
    pipeline.

    ```python exec="true" source="material-block" html="true"
    from amltk.pipeline import Sequential, Choice, Node, factorize

    pipeline = Sequential(
        Choice(Node(name="hi"), Node(name="hello"), name="choice"),
        Node(name="banana"),
        name="pipeline",
    )

    from amltk._doc import doc_print; _print = print; print = lambda thing: doc_print(_print, thing)  # markdown-exec: hide
    print(pipeline)
    for i, possibility in enumerate(factorize(pipeline)):
        print(f"Pipeline {i}:")
        print(possibility)
    ```

    Args:
        node: The node to factorize.
        factor_by: A function that takes a node and returns True if it
            should be factorized into its children, False otherwise. By
            default, it will split only Choice nodes.
        assign_child: A function that takes a node and a child and
            returns a new node with that child assigned to it. By default,
            it will mutate the node so that it has that child as its
            only child. You may wish to pass in custom functionality if there
            is more than one way to assign a child to a node or extra logic must
            be done to the nodes properties.

            It should return the same type of node as the one passed in.

    Returns:
        An iterator over all possible pipelines.
    """  # noqa: E501
    # NOTE: These two functions below are defined here instead to allow custom
    # Node types in the future. The default behaviour is defined to just split
    # Choice nodes and assign a child to one is to just mutate the node so
    # that it has that child as its only child.
    if factor_by is None:
        factor_by = lambda _node: isinstance(_node, Choice)

    if assign_child is None:
        assign_child = lambda _node, _child: _node.mutate(nodes=(_child,))

    _factorize = partial(factorize, factor_by=factor_by, assign_child=assign_child)

    match node:
        case Node(nodes=()):
            # Base case, there's no further possibility to factorize
            yield node.copy()
        case Node(nodes=children) if factor_by(node):
            for child in children:
                for possible_child in _factorize(child):
                    split_node_with_child_assigned = assign_child(node, possible_child)  # type: ignore
                    yield split_node_with_child_assigned  # type: ignore

        case Node(nodes=children):
            # We need to return N copies of this node, with each
            # enumerating over all the posibilities of its children
            # e.g.
            # | children_sets = ((1, 2), (3, 4), (5,))
            # | for child_set in [(1, 3, 5,), (1, 4, 5,), (2, 3, 5,), (2, 4, 5,)]:
            # |    yield node.mutate(nodes=child_set)
            children_sets = (_factorize(c) for c in children)
            for child_set in product(*children_sets):
                yield node.mutate(nodes=child_set)
