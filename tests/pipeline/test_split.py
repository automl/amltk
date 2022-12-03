from byop.pipeline import split, step
from byop.pipeline.step import Step


def test_split() -> None:
    """Test splitting a pipeline into two"""
    split_step = split(
        "split",
        step("1", 1) | step("2", 2),
        step("3", 3) | step("4", 4),
    )
    assert len(split_step.paths) == 2


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
    s1, s2, s3, s4, s5, s6, s7, s8 = [step(str(i), i) for i in range(1, 9)]
    split1 = split("split1", s1, s2)
    split2 = split("split1", s3, s4)
    split3 = split("split1", s5, s6)
    split4 = split("split1", s7, s8)
    steps = Step.join(split1, split2, split3, split4)

    expected = [split1, s1, s2, split2, s3, s4, split3, s5, s6, split4, s7, s8]
    assert list(steps.traverse()) == expected


def test_traverse_deep() -> None:
    s1, s2, s3, s4, s5, s6, s7, s8 = [step(str(i), i) for i in range(1, 9)]
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
