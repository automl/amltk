from __future__ import annotations

from amltk.pipeline import Choice, Node
from amltk.pipeline.components import Sequential
from amltk.pipeline.ops import factorize


def test_factorize_base_case():
    node = Node(name="hi")
    assert list(factorize(node)) == [node]


def test_factorize_with_no_choices_returns_same_pipeline():
    node = Node(Node(name="n1"), Node(name="n2"), name="n0")
    assert list(factorize(node)) == [node]


def test_factorize_with_single_choice_same_pipeline():
    node = Choice(Node(name="n1"), name="c1")
    assert list(factorize(node)) == [node]


def test_factorize_with_two_choices_returns_both_pipelines():
    n1 = Node(name="n1")
    n2 = Node(name="n2")
    choice = Choice(n1, n2, name="c1")
    expected = [
        Choice(n1, name="c1"),
        Choice(n2, name="c1"),
    ]
    assert list(factorize(choice)) == expected


def test_nested_choice_returns_possible_pipelines():
    n1 = Node(name="n1")
    n2 = Node(name="n2")
    n3 = Node(name="n3")
    choice = Choice(n1, n2, name="c1")
    top = Sequential(choice, n3, name="s1")
    expected = [
        Sequential(Choice(n1, name="c1"), n3, name="s1"),
        Sequential(Choice(n2, name="c1"), n3, name="s1"),
    ]
    assert list(factorize(top)) == expected


def test_choice_followed_by_choice():
    n1 = Node(name="n1")
    n2 = Node(name="n2")
    n3 = Node(name="n3")
    pipeline = Sequential(
        Choice(Choice(n1, n2, name="c2"), n3, name="c1"),
        name="s1",
    )
    expected = [
        Sequential(
            Choice(
                Choice(n1, name="c2"),
                name="c1",
            ),
            name="s1",
        ),
        Sequential(
            Choice(
                Choice(n2, name="c2"),
                name="c1",
            ),
            name="s1",
        ),
        Sequential(
            Choice(n3, name="c1"),
            name="s1",
        ),
    ]
    assert list(factorize(pipeline)) == expected


def test_double_nested_choice():
    S = Sequential
    C = Choice
    N = Node
    pipeline = S(
        S(
            C(N(name="n3.1.1"), N(name="n3.1.2"), name="c3.1"),
            C(N(name="n3.2.1"), N(name="n3.2.2"), name="c3.2"),
            name="s2.1",
        ),
        N(name="n2.2"),
        name="s1",
    )
    expected = [
        S(
            S(
                C(N(name="n3.1.1"), name="c3.1"),
                C(N(name="n3.2.1"), name="c3.2"),
                name="s2.1",
            ),
            N(name="n2.2"),
            name="s1",
        ),
        S(
            S(
                C(N(name="n3.1.1"), name="c3.1"),
                C(N(name="n3.2.2"), name="c3.2"),
                name="s2.1",
            ),
            N(name="n2.2"),
            name="s1",
        ),
        S(
            S(
                C(N(name="n3.1.2"), name="c3.1"),
                C(N(name="n3.2.1"), name="c3.2"),
                name="s2.1",
            ),
            N(name="n2.2"),
            name="s1",
        ),
        S(
            S(
                C(N(name="n3.1.2"), name="c3.1"),
                C(N(name="n3.2.2"), name="c3.2"),
                name="s2.1",
            ),
            N(name="n2.2"),
            name="s1",
        ),
    ]

    assert list(factorize(pipeline)) == expected


def test_factorize_with_min_depth_triggers_if_at_min_depth():
    n1 = Node(name="n1")
    n2 = Node(name="n2")

    # Choice is at depth 1
    seq = Sequential(Choice(n1, n2, name="c1"), name="s1")

    expected = [
        Sequential(Choice(n1, name="c1"), name="s1"),
        Sequential(Choice(n2, name="c1"), name="s1"),
    ]
    assert list(factorize(seq, min_depth=1)) == expected


def test_factorize_with_min_depth_greater_does_not_factor():
    n1 = Node(name="n1")
    n2 = Node(name="n2")

    # Choice is at depth 0
    choice = Choice(n1, n2, name="c1")

    assert list(factorize(choice, min_depth=1_000)) == [choice]


def test_factorize_with_max_depth_does_not_triggers_if_past_max_depth():
    S = Sequential
    n1 = Node(name="n1")
    n2 = Node(name="n2")

    # Choice is at depth 2
    seq = S(S(Choice(n1, n2, name="c1"), name="s1"), name="s2")
    assert list(factorize(seq, max_depth=1)) == [seq]


def test_factorize_with_max_depth_triggers_if_not_at_max_depth():
    n1 = Node(name="n1")
    n2 = Node(name="n2")

    # Choice is at depth 2
    seq = Sequential(Choice(n1, n2, name="c1"), name="s1")

    expected = [
        Sequential(Choice(n1, name="c1"), name="s1"),
        Sequential(Choice(n2, name="c1"), name="s1"),
    ]
    assert list(factorize(seq, max_depth=1_000)) == expected


def test_with_custom_factor_by():
    # We'll factorize by the name of the node
    factor_by = lambda _node: _node.name == "estimator"

    e1 = Node(name="e1")
    e2 = Node(name="e2")
    estimator = Choice(e1, e2, name="estimator")

    p1 = Node(name="p1")
    p2 = Node(name="p2")
    preprocessor = Choice(p1, p2, name="preprocessor")

    pipeline = Sequential(preprocessor, estimator, name="pipeline")

    # Should only split out the estimator
    expected = [
        Sequential(
            Choice(p1, p2, name="preprocessor"),
            Choice(e1, name="estimator"),
            name="pipeline",
        ),
        Sequential(
            Choice(p1, p2, name="preprocessor"),
            Choice(e2, name="estimator"),
            name="pipeline",
        ),
    ]
    assert list(factorize(pipeline, factor_by=factor_by)) == expected
