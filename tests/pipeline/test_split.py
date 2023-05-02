from __future__ import annotations

from byop import split, step
from byop.pipeline import Step


def test_split() -> None:
    split_step = split(
        "split",
        step("1", 1) | step("2", 2),
        step("3", 3) | step("4", 4),
    )
    expected_npaths = 2
    assert len(split_step.paths) == expected_npaths


def test_traverse_one_layer() -> None:
    s1, s2, s3, s4 = step("1", 1), step("2", 2), step("3", 3), step("4", 4)
    split_step = split("split", s1 | s2, s3 | s4)

    assert list(split_step.traverse()) == [split_step, s1, s2, s3, s4]


def test_traverse_one_deep() -> None:
    s1, s2, s3, s4 = step("1", 1), step("2", 2), step("3", 3), step("4", 4)
    subsplit = split("subsplit", s3 | s4)
    split_step = split("split", s1, s2 | subsplit)

    assert list(split_step.traverse()) == [split_step, s1, s2, subsplit, s3, s4]


def test_traverse_sequential_splits() -> None:
    s1, s2, s3, s4, s5, s6, s7, s8 = (step(str(i), i) for i in range(1, 9))
    split1 = split("split1", s1, s2)
    split2 = split("split2", s3, s4)
    split3 = split("split3", s5, s6)
    split4 = split("split4", s7, s8)
    steps = Step.join(split1, split2, split3, split4)

    expected = [split1, s1, s2, split2, s3, s4, split3, s5, s6, split4, s7, s8]
    assert list(steps.traverse()) == expected


def test_traverse_deep() -> None:
    s1, s2, s3, s4, s5, s6, s7, s8 = (step(str(i), i) for i in range(1, 9))
    subsub_split1 = split("subsplit1", s3 | s4)
    sub_split1 = split("subsubsplit1", s1, s2 | subsub_split1)

    subsub_split2 = split("subsplit2", s7 | s8)
    sub_split2 = split("subssubplit2", s5, s6 | subsub_split2)

    split_step = split("split1", sub_split1, sub_split2)

    expected = [
        split_step,
        sub_split1,
        s1,
        s2,
        subsub_split1,
        s3,
        s4,
        sub_split2,
        s5,
        s6,
        subsub_split2,
        s7,
        s8,
    ]
    assert list(split_step.traverse()) == expected


def test_remove_split() -> None:
    s1, s2, s3, s4, s5 = (
        step("1", 1),
        step("2", 2),
        step("3", 3),
        step("4", 4),
        step("5", 5),
    )
    split_step = split(
        "split",
        s1,
        s2 | split("subsplit", s3 | s4) | s5,
    )

    new = Step.join(split_step.remove(["subsplit"]))
    assert new == split(
        "split",
        s1,
        s2 | s5,
    )

    new = Step.join(split_step.remove(["3"]))
    assert new == split(
        "split",
        s1,
        s2 | split("subsplit", s4) | s5,
    )


def test_replace_split() -> None:
    s1, s2, s3, s4, s5 = (
        step("1", 1),
        step("2", 2),
        step("3", 3),
        step("4", 4),
        step("5", 5),
    )
    split_step = split(
        "split",
        s1,
        s2 | split("subsplit", s3 | s4) | s5,
    )

    replacement = step("replacement", 0)
    new = Step.join(split_step.replace({"subsplit": replacement}))
    assert new == split(
        "split",
        s1,
        s2 | replacement | s5,
    )

    new = Step.join(split_step.replace({"3": replacement}))
    assert new == split(
        "split",
        s1,
        s2 | split("subsplit", replacement | s4) | s5,
    )


def test_split_on_path_with_one_entry_removes_properly() -> None:
    s = split("split", step("1", 1), step("2", 2))
    result = next(s.remove(["1"]))
    assert result == split("split", step("2", 2))


def test_split_on_head_of_path_does_not_remove_rest_of_path() -> None:
    s = split("split", step("1", 1) | step("2", 2))
    result = next(s.remove(["1"]))
    assert result == split("split", step("2", 2))


def test_configure_single() -> None:
    s1 = split(
        "split",
        step("1", 1, space={"a": [1, 2, 3]}) | step("2", 2, space={"b": [1, 2, 3]}),
        step("3", 3, space={"c": [1, 2, 3]}),
        item=object(),
        space={"split_space": [1, 2, 3]},
    )
    configured_s1 = s1.configure({"split_space": 1, "1:a": 1, "2:b": 2, "3:c": 3})

    expected_configs_by_name = {
        "split": {"split_space": 1},
        "1": {"a": 1},
        "2": {"b": 2},
        "3": {"c": 3},
    }
    for s in configured_s1.traverse():
        assert s.config == expected_configs_by_name[s.name]
        assert s.search_space is None


def test_configure_chained() -> None:
    head = (
        split(
            "split",
            step("1", 2, space={"a": [1, 2, 3]}),
        )
        | step("2", 1, space={"b": [1, 2, 3]})
        | step("3", 3, space={"c": [1, 2, 3]})
    )
    configured_head = head.configure({"split:1:a": 1, "2:b": 2, "3:c": 3})

    expected_configs = {
        "split": None,
        "1": {"a": 1},
        "2": {"b": 2},
        "3": {"c": 3},
    }
    for s in configured_head.traverse():
        assert s.config == expected_configs[s.name]
        assert s.search_space is None


def test_qualified_name() -> None:
    head = split(
        "split",
        step("1", 2),
        split("subsplit", step("2", 1) | step("3", 3)),
    )
    assert head.qualified_name() == "split"

    s1 = head.find("1")
    assert s1 is not None
    assert s1.qualified_name() == "split:1"

    subsplit = head.find("subsplit")
    assert subsplit is not None
    assert subsplit.qualified_name() == "split:subsplit"

    s2 = head.find("2")
    assert s2 is not None
    assert s2.qualified_name() == "split:subsplit:2"

    s3 = head.find("3")
    assert s3 is not None
    assert s3.qualified_name() == "split:subsplit:3"


def test_path_to() -> None:
    head = split(
        "split",
        step("1", 2),
        split(
            "subsplit",
            step("2", 1) | step("3", 3),
        ),
    )
    _split = head.find("split")
    assert _split is not None

    s1 = head.find("1")
    assert s1 is not None

    subsplit = head.find("subsplit")
    assert subsplit is not None

    s2 = head.find("2")
    assert s2 is not None

    s3 = head.find("3")
    assert s3 is not None

    assert _split.path_to(_split) == [_split]
    assert _split.path_to(s1) == [_split, s1]
    assert _split.path_to(subsplit) == [_split, subsplit]
    assert _split.path_to(s2) == [_split, subsplit, s2]
    assert _split.path_to(s3) == [_split, subsplit, s2, s3]

    assert s1.path_to(_split) == [s1, _split]
    assert s1.path_to(s1) == [s1]
    assert s1.path_to(subsplit) is None
    assert s1.path_to(s2) is None
    assert s1.path_to(s3) is None

    assert subsplit.path_to(_split) == [subsplit, _split]
    assert subsplit.path_to(s1) is None
    assert subsplit.path_to(subsplit) == [subsplit]
    assert subsplit.path_to(s2) == [subsplit, s2]
    assert subsplit.path_to(s3) == [subsplit, s2, s3]

    assert s2.path_to(_split) == [s2, subsplit, _split]
    assert s2.path_to(s1) is None
    assert s2.path_to(subsplit) == [s2, subsplit]
    assert s2.path_to(s2) == [s2]
    assert s2.path_to(s3) == [s2, s3]

    assert s3.path_to(_split) == [s3, s2, subsplit, _split]
    assert s3.path_to(s1) is None
    assert s3.path_to(subsplit) == [s3, s2, subsplit]
    assert s3.path_to(s2) == [s3, s2]
    assert s3.path_to(s3) == [s3]
