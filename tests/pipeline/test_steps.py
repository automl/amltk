from __future__ import annotations

from amltk.pipeline import Component, Step, request, step


def test_step_component() -> None:
    o = object()
    s = step("name", o)
    assert s.name == "name"
    assert s.item == o
    assert s.config is None
    assert isinstance(s, Component)


def test_step_searchable() -> None:
    s = step("name", object(), space={"a": [1, 2]}, config={"b": 2})
    assert s.name == "name"
    assert s.search_space == {"a": [1, 2]}
    assert s.config == {"b": 2}
    assert isinstance(s, Component)


def test_step_joinable() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    steps = s1 | s2

    assert list(steps.iter()) == [s1, s2]


def test_step_head() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)

    x = s1 | s2

    assert x.head() == s1


def test_step_tail() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)

    x = s1 | s2

    assert x.tail() == s2


def test_step_iter() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    s3 = step("3", 3)
    start = s1 | s2 | s3

    middle = start.nxt
    assert middle is not None
    end = middle.nxt
    assert end is not None

    # start - middle - end
    # s1    - s2     - s3

    assert list(start.iter()) == [s1, s2, s3]
    assert list(middle.iter()) == [s2, s3]
    assert list(end.iter()) == [s3]

    assert list(start.iter(backwards=True)) == [s1]
    assert list(middle.iter(backwards=True)) == [s2, s1]
    assert list(end.iter(backwards=True)) == [s3, s2, s1]

    assert list(start.proceeding()) == [s2, s3]
    assert list(middle.proceeding()) == [s3]
    assert list(end.proceeding()) == []

    assert list(start.preceeding()) == []
    assert list(middle.preceeding()) == [s1]
    assert list(end.preceeding()) == [s1, s2]


def test_join() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    s3 = step("3", 3)

    assert list(Step.join([s1, s2, s3]).iter()) == [s1, s2, s3]
    assert list(Step.join(s1, [s2, s3]).iter()) == [s1, s2, s3]
    assert list(Step.join([s1], [s2, s3]).iter()) == [s1, s2, s3]
    assert list(Step.join([s1, s2], s3).iter()) == [s1, s2, s3]


def test_append_single() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    x = s1.append(s2)

    assert list(x.iter()) == [s1, s2]


def test_append_chain() -> None:
    s1 = step("1", 1)
    s2 = step("2", 2)
    s3 = step("3", 3)
    x = s1.append(s2 | s3)

    assert list(x.iter()) == [s1, s2, s3]


def test_configure_single() -> None:
    s1 = step("1", 1, space={"a": [1, 2, 3]})
    configured_s1 = s1.configure({"a": 1})

    assert configured_s1.config == {"a": 1}
    assert configured_s1.search_space is None


def test_configure_chain() -> None:
    head = (
        step("1", 1, space={"a": [1, 2, 3]})
        | step("2", 2, space={"b": [1, 2, 3]})
        | step("3", 3, space={"c": [1, 2, 3]})
    )
    configured_head = head.configure({"1:a": 1, "2:b": 2, "3:c": 3})

    expected_configs = [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]
    for s, expected_config in zip(configured_head.iter(), expected_configs):
        assert s.config == expected_config
        assert s.search_space is None


def test_qualified_name() -> None:
    head = step("1", 1) | step("2", 2) | step("3", 3)
    last = head.tail()

    # Should not have any prefixes from the other steps
    assert last.qualified_name() == "3"


def test_path_to() -> None:
    head = step("1", 1) | step("2", 2) | step("3", 3)

    s1 = head.find("1")
    assert s1 is not None

    s2 = head.find("2")
    assert s2 is not None

    s3 = head.find("3")
    assert s3 is not None

    assert s1.path_to(s1) == [s1]
    assert s1.path_to(s3) == [s1, s2, s3]
    assert s1.path_to(s2) == [s1, s2]

    assert s2.path_to(s2) == [s2]
    assert s2.path_to(s3) == [s2, s3]
    assert s2.path_to(s1) == [s2, s1]

    assert s3.path_to(s3) == [s3]
    assert s3.path_to(s1) == [s3, s2, s1]
    assert s3.path_to(s2) == [s3, s2]

    assert s3.path_to(s1, direction="forward") is None
    assert s2.path_to(s1, direction="forward") is None
    assert s1.path_to(s1, direction="forward") == [s1]

    assert s3.path_to(s3, direction="backward") == [s3]
    assert s2.path_to(s3, direction="backward") is None
    assert s1.path_to(s3, direction="backward") is None


def test_param_request() -> None:
    component = step(
        "rf",
        1,
        space={"n_estimators": (10, 100), "criterion": ["gini", "entropy"]},
        config={"random_state": request("seed", default=None)},
    )

    config = {"n_estimators": 10, "criterion": "gini"}
    configured_component = component.configure(config, params={"seed": 42})

    assert configured_component == step(
        "rf",
        1,
        config={
            "n_estimators": 10,
            "criterion": "gini",
            "random_state": 42,
        },
    )
