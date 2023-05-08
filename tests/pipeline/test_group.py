from __future__ import annotations

from byop.pipeline import group, step


def test_group_path_to_simple() -> None:
    g = group("group", step("a", 1) | step("b", 2) | step("c", 3))

    a = g.find("a")
    b = g.find("b")
    c = g.find("c")

    assert g.path_to("group") == [g]
    assert g.path_to("a") == [g, a]
    assert g.path_to("b") == [g, a, b]
    assert g.path_to("c") == [g, a, b, c]


def test_group_path_to_deep() -> None:
    g = (
        step("head", "head")
        | group(
            "group",
            step("a", 1),
            step("b", 2),
            group("group2", step("c", 3), step("d", 4)) | step("e", 5),
        )
        | step("tail", "tail")
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
    g = group("group", step("a", 1), step("b", 2), step("c", 3))

    assert next(g.select({"group": "a"})) == step("a", 1)
    assert next(g.select({"group": "b"})) == step("b", 2)
    assert next(g.select({"group": "c"})) == step("c", 3)


def test_group_deep_select() -> None:
    g = (
        step("head", "head")
        | group(
            "group",
            step("a", 1),
            step("b", 2),
            group("group2", step("c", 3), step("d", 4)) | step("e", 5),
        )
        | step("tail", "tail")
    )

    expected = step("head", "head") | step("d", 4) | step("e", 5) | step("tail", "tail")

    chosen = next(g.select({"group": "group2", "group2": "d"}))
    assert chosen == expected


def test_group_simple_traverse() -> None:
    g = group("group", step("a", 1), step("b", 2), step("c", 3))

    assert list(g.traverse()) == [g, step("a", 1), step("b", 2), step("c", 3)]


def test_group_deep_traverse() -> None:
    g = (
        step("head", "head")
        | group(
            "group",
            step("a", 1),
            step("b", 2),
            group("group2", step("c", 3), step("d", 4)) | step("e", 5),
        )
        | step("tail", "tail")
    )

    _group = g.find("group")
    group2 = g.find("group2")

    expected = [
        step("head", "head"),
        _group,
        step("a", 1),
        step("b", 2),
        group2,
        step("c", 3),
        step("d", 4),
        step("e", 5),
        step("tail", "tail"),
    ]

    assert list(g.traverse()) == expected


def test_group_simple_walk() -> None:
    g = group("group", step("a", 1), step("b", 2), step("c", 3))

    assert list(g.walk()) == [
        ([], [], g),
        ([g], [], step("a", 1)),
        ([g], [], step("b", 2)),
        ([g], [], step("c", 3)),
    ]


def test_group_list_walk() -> None:
    g = group("group", step("a", 1) | step("b", 2) | step("c", 3))

    assert list(g.walk()) == [
        ([], [], g),
        ([g], [], step("a", 1)),
        ([g], [step("a", 1)], step("b", 2)),
        ([g], [step("a", 1), step("b", 2)], step("c", 3)),
    ]


def test_group_deep_walk() -> None:
    g = (
        step("head", "head")
        | group(
            "group",
            step("a", 1),
            step("b", 2),
            group(
                "group2",
                step("c", 3),
                step("d", 4) | step("extra", 15),
            )
            | step("e", 5),
        )
        | step("tail", "tail")
    )

    _group = g.find("group")
    group2 = g.find("group2")
    head = g.find("head")

    expected = [
        ([], [], head),
        ([], [head], _group),
        ([_group], [], step("a", 1)),
        ([_group], [], step("b", 2)),
        ([_group], [], group2),
        ([_group, group2], [], step("c", 3)),
        ([_group, group2], [], step("d", 4)),
        ([_group, group2], [step("d", 4)], step("extra", 15)),
        ([_group], [group2], step("e", 5)),
        ([], [head, _group], step("tail", "tail")),
    ]

    assert list(g.walk()) == expected


def test_group_configure_simple() -> None:
    g = group(
        "group",
        step("a", 1, space={"hp": [1, 2, 3]}),
        step("b", 2, space={"hp": [4, 5, 6]}),
        step("c", 3, space={"hp": [7, 8, 9]}),
    )
    expected = group(
        "group",
        step("a", 1, config={"hp": 1}),
        step("b", 2, config={"hp": 4}),
        step("c", 3, config={"hp": 7}),
    )

    configuration = {
        "group:a:hp": 1,
        "group:b:hp": 4,
        "group:c:hp": 7,
    }

    assert g.configure(configuration) == expected


def test_group_configure_deep() -> None:
    g = (
        step("head", "head")
        | group(
            "group",
            step("a", 1, space={"hp": [1, 2, 3]}),
            step("b", 2, space={"hp": [4, 5, 6]}),
            group(
                "group2",
                step("c", 3, space={"hp": [7, 8, 9]}),
                step("d", 4, space={"hp": [10, 11, 12]})
                | step("extra", 15, space={"hp": [21, 22, 23]}),
            )
            | step("e", 5, space={"hp": [13, 14, 15]}),
        )
        | step("tail", "tail")
    )
    expected = (
        step("head", "head")
        | group(
            "group",
            step("a", 1, config={"hp": 1}),
            step("b", 2, config={"hp": 4}),
            group(
                "group2",
                step("c", 3, config={"hp": 7}),
                step("d", 4, config={"hp": 10}) | step("extra", 15, config={"hp": 21}),
            )
            | step("e", 5, config={"hp": 13}),
        )
        | step("tail", "tail")
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
