from byop.functional import reposition


def test_reposition() -> None:
    xs = [3, 2, 1]
    assert reposition(xs, [1, ...]) == [1, 3, 2]
    assert reposition(xs, [..., 3]) == [2, 1, 3]
    assert reposition(xs, [1, ..., 3]) == [1, 2, 3]

    assert reposition(list(range(100)), [98, ...]) == [98, *list(range(98)), 99]

    xs = [1, 2, 3, 4, 5, 6]
    assert reposition(xs, [6, 5, ..., 2, 1]) == [6, 5, 3, 4, 2, 1]
