from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from amltk.exceptions import RequestNotMetError
from amltk.pipeline import Choice, Join, Node, Sequential, request


def test_node_rshift() -> None:
    node1 = Node(name="n1")
    node2 = Node(name="n2")
    node3 = Node(name="n3")

    out = node1 >> node2 >> node3
    expected_nodes = (node1, node2, node3)

    assert isinstance(out, Sequential)
    assert out.nodes == expected_nodes
    for a, b in zip(out.nodes, expected_nodes, strict=True):
        assert id(a) != id(b)


def test_node_and() -> None:
    node1 = Node(name="node1")
    node2 = Node(name="node2")
    node3 = Node(name="node3")

    out = node1 & node2 & node3
    expected_nodes = (node1, node2, node3)

    assert isinstance(out, Join)
    assert out.nodes == expected_nodes
    for a, b in zip(out.nodes, expected_nodes, strict=True):
        assert id(a) != id(b)


def test_node_or() -> None:
    node1 = Node(name="node1")
    node2 = Node(name="node2")
    node3 = Node(name="node3")

    out = node1 | node2 | node3
    expected_nodes = (node1, node2, node3)

    assert isinstance(out, Choice)
    assert set(out.nodes) == set(expected_nodes)
    for a, b in zip(out.nodes, expected_nodes, strict=True):
        assert id(a) != id(b)


def test_single_node_configure() -> None:
    node = Node(name="node")
    node = node.configure({"a": 1, "b": 2})
    assert node == Node(name="node", config={"a": 1, "b": 2})

    node = Node(name="node")
    node = node.configure({"node:a": 1, "node:b": 2})
    assert node == Node(name="node", config={"a": 1, "b": 2})


def test_with_children_configure() -> None:
    node = Node(Node(name="child1"), Node(name="child2"), name="node")
    node = node.configure(
        {"node:a": 1, "node:b": 2, "node:child1:c": 3, "node:child2:d": 4},
    )

    assert node == Node(
        Node(config={"c": 3}, name="child1"),
        Node(config={"d": 4}, name="child2"),
        name="node",
        config={"a": 1, "b": 2},
    )


def test_deeply_nested_children_configuration() -> None:
    node = Node(
        Node(Node(name="child2"), name="child1"),
        name="node",
    )
    node = node.configure({"node:a": 1, "node:child1:b": 2, "node:child1:child2:c": 3})

    assert node == Node(
        Node(
            Node(name="child2", config={"c": 3}),
            name="child1",
            config={"b": 2},
        ),
        name="node",
        config={"a": 1},
    )


def test_configure_with_transform() -> None:
    def _transform(config: Mapping[str, Any], _) -> dict:
        c = (config["a"], config["b"])
        return {"c": c}

    node = Node(config_transform=_transform, name="1")
    node = node.configure({"1:a": 1, "1:b": 2})
    assert node == Node(config={"c": (1, 2)}, name="1", config_transform=_transform)


def test_configure_with_param_request() -> None:
    node = Node(
        config={
            "x": request("x"),
            "y": request("y"),
            "z": request("z", default=3),
        },
        name="1",
    )

    # Should configure as expected, with default and specified values
    conf_node = node.configure({"a": -1}, params={"x": 1, "y": 2})
    assert conf_node == Node(name="1", config={"a": -1, "x": 1, "y": 2, "z": 3})

    # When trying to configure with "x" missing, should raise
    with pytest.raises(RequestNotMetError):
        node.configure({"a": -1}, params={"y": 2})


def test_find() -> None:
    n1 = Node(name="1")
    n2 = Node(name="2")
    n3 = Node(name="3")
    seq = n1 >> n2 >> n3

    s1 = seq.find("1")
    assert s1 is seq.nodes[0]

    s2 = seq.find("2")
    assert s2 is seq.nodes[1]

    s3 = seq.find("3")
    assert s3 is seq.nodes[2]

    s4 = seq.find("4")
    assert s4 is None

    default = Node(name="default")
    s5 = seq.find("5", default=default)
    assert s5 is default


def test_walk() -> None:
    n1 = Node(name="1")

    sub3 = Node(name="sub3")
    sub2 = Node(sub3, name="sub2")
    n2 = Node(sub2, name="2")

    n3 = Node(name="3")

    seq = n1 >> n2 >> n3

    expected_path = [
        ([], seq),
        ([seq], n1),
        ([seq, n1], n2),
        ([seq, n1, n2], sub2),
        ([seq, n1, n2, sub2], sub3),
        ([seq, n1, n2], n3),
    ]

    for (path, node), (_exp_path, _exp_node) in zip(
        seq.walk(),
        expected_path,
        strict=True,
    ):
        assert node == _exp_node
        assert path == _exp_path
