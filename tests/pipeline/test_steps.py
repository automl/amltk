from byop.pipeline.api import step
from byop.pipeline.components import Component, Searchable, Step


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
    assert s.space == {"a": [1, 2]}
    assert s.config == {"b": 2}
    assert isinstance(s, Searchable)


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
    assert list(end.preceeding()) == [s2, s1]


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
