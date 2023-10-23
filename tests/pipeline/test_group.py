from __future__ import annotations

from amltk.pipeline import group, step


def test_group_path_to_simple() -> None:
    g = group("group", step("a", object) | step("b", object) | step("c", object))

    a = g.find("a")
    b = g.find("b")
    c = g.find("c")

    assert g.path_to("group") == [g]
    assert g.path_to("a") == [g, a]
    assert g.path_to("b") == [g, a, b]
    assert g.path_to("c") == [g, a, b, c]


def test_group_path_to_deep() -> None:
    g = (
        step("head", object)
        | group(
            "group",
            step("a", object),
            step("b", object),
            group("group2", step("c", object), step("d", object)) | step("e", object),
        )
        | step("tail", object)
    )

    head = g.find("head")
    _group = g.find("group")
    a = g.find("a")
    b = g.find("b")
    group2 = g.find("group2")
    c = g.find("c")
    d = g.find("d")
    e = g.find("e")
    tail = g.find("tail")

    assert g.path_to("head") == [head]
    assert g.path_to("group") == [head, _group]
    assert g.path_to("a") == [head, _group, a]
    assert g.path_to("b") == [head, _group, b]
    assert g.path_to("group2") == [head, _group, group2]
    assert g.path_to("c") == [head, _group, group2, c]
    assert g.path_to("d") == [head, _group, group2, d]
    assert g.path_to("e") == [head, _group, group2, e]
    assert g.path_to("tail") == [head, _group, tail]


def test_group_simple_select() -> None:
    g = group("group", step("a", object), step("b", object), step("c", object))

    assert next(g.select({"group": "a"})) == step("a", object)
    assert next(g.select({"group": "b"})) == step("b", object)
    assert next(g.select({"group": "c"})) == step("c", object)


def test_group_deep_select() -> None:
    g = (
        step("head", object)
        | group(
            "group",
            step("a", object),
            step("b", object),
            group("group2", step("c", object), step("d", object)) | step("e", object),
        )
        | step("tail", object)
    )

    expected = (
        step("head", object)
        | step("d", object)
        | step("e", object)
        | step("tail", object)
    )

    chosen = next(g.select({"group": "group2", "group2": "d"}))
    assert chosen == expected


def test_group_simple_traverse() -> None:
    g = group("group", step("a", object), step("b", object), step("c", object))

    assert list(g.traverse()) == [
        g,
        step("a", object),
        step("b", object),
        step("c", object),
    ]


def test_group_deep_traverse() -> None:
    g = (
        step("head", object)
        | group(
            "group",
            step("a", object),
            step("b", object),
            group("group2", step("c", object), step("d", object)) | step("e", object),
        )
        | step("tail", object)
    )

    _group = g.find("group")
    group2 = g.find("group2")

    expected = [
        step("head", object),
        _group,
        step("a", object),
        step("b", object),
        group2,
        step("c", object),
        step("d", object),
        step("e", object),
        step("tail", object),
    ]

    assert list(g.traverse()) == expected


def test_group_simple_walk() -> None:
    g = group("group", step("a", object), step("b", object), step("c", object))

    assert list(g.walk()) == [
        ([], [], g),
        ([g], [], step("a", object)),
        ([g], [], step("b", object)),
        ([g], [], step("c", object)),
    ]


def test_group_list_walk() -> None:
    g = group("group", step("a", object) | step("b", object) | step("c", object))

    assert list(g.walk()) == [
        ([], [], g),
        ([g], [], step("a", object)),
        ([g], [step("a", object)], step("b", object)),
        ([g], [step("a", object), step("b", object)], step("c", object)),
    ]


def test_group_deep_walk() -> None:
    g = (
        step("head", object)
        | group(
            "group",
            step("a", object),
            step("b", object),
            group(
                "group2",
                step("c", object),
                step("d", object) | step("extra", object),
            )
            | step("e", object),
        )
        | step("tail", object)
    )

    _group = g.find("group")
    group2 = g.find("group2")
    head = g.find("head")

    expected = [
        ([], [], head),
        ([], [head], _group),
        ([_group], [], step("a", object)),
        ([_group], [], step("b", object)),
        ([_group], [], group2),
        ([_group, group2], [], step("c", object)),
        ([_group, group2], [], step("d", object)),
        ([_group, group2], [step("d", object)], step("extra", object)),
        ([_group], [group2], step("e", object)),
        ([], [head, _group], step("tail", object)),
    ]

    assert list(g.walk()) == expected


def test_group_configure_simple() -> None:
    g = group(
        "group",
        step("a", object, space={"hp": [1, 2, 3]}),
        step("b", object, space={"hp": [4, 5, 6]}),
        step("c", object, space={"hp": [7, 8, 9]}),
    )
    expected = group(
        "group",
        step("a", object, config={"hp": 1}),
        step("b", object, config={"hp": 4}),
        step("c", object, config={"hp": 7}),
    )

    configuration = {
        "group:a:hp": 1,
        "group:b:hp": 4,
        "group:c:hp": 7,
    }

    assert g.configure(configuration) == expected


def test_group_configure_deep() -> None:
    g = (
        step("head", object)
        | group(
            "group",
            step("a", object, space={"hp": [1, 2, 3]}),
            step("b", object, space={"hp": [4, 5, 6]}),
            group(
                "group2",
                step("c", object, space={"hp": [7, 8, 9]}),
                step("d", object, space={"hp": [10, 11, 12]})
                | step("extra", object, space={"hp": [21, 22, 23]}),
            )
            | step("e", object, space={"hp": [13, 14, 15]}),
        )
        | step("tail", object)
    )
    expected = (
        step("head", object)
        | group(
            "group",
            step("a", object, config={"hp": 1}),
            step("b", object, config={"hp": 4}),
            group(
                "group2",
                step("c", object, config={"hp": 7}),
                step("d", object, config={"hp": 10})
                | step("extra", object, config={"hp": 21}),
            )
            | step("e", object, config={"hp": 13}),
        )
        | step("tail", object)
    )

    config = {
        "group:a:hp": 1,
        "group:b:hp": 4,
        "group:group2:c:hp": 7,
        "group:group2:d:hp": 10,
        "group:group2:d:extra:hp": 21,
        "group:e:hp": 13,
    }

    assert g.configure(config) == expected
