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
    split2 = split("split1", s3, s4)
    split3 = split("split1", s5, s6)
    split4 = split("split1", s7, s8)
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
