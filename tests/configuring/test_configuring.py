from __future__ import annotations

from byop import Pipeline, choice, searchable, split, step


def test_heirarchical_str() -> None:
    pipeline = Pipeline.create(
        step("one", 1, space={"v": [1, 2, 3]}),
        split(
            "split",
            step("x", 1, space={"v": [4, 5, 6]}),
            step("y", 1, space={"v": [4, 5, 6]}),
        ),
        choice(
            "choice",
            step("a", 1, space={"v": [4, 5, 6]}),
            step("b", 1, space={"v": [4, 5, 6]}),
        ),
    )
    config = {
        "one:v": 1,
        "split:x:v": 4,
        "split:y:v": 5,
        "choice": "a",
        "choice:a:v": 6,
    }
    result = pipeline.configure(config)

    expected = Pipeline.create(
        step("one", 1, config={"v": 1}),
        split(
            "split",
            step("x", 1, config={"v": 4}),
            step("y", 1, config={"v": 5}),
        ),
        step("a", 1, config={"v": 6}),
        name=pipeline.name,
    )

    assert result == expected


def test_heirarchical_str_with_predefined_configs() -> None:
    pipeline = Pipeline.create(
        step("one", 1, config={"v": 1}),
        split(
            "split",
            step("x", 1),
            step("y", 1, space={"v": [4, 5, 6]}),
        ),
        choice(
            "choice",
            step("a", 1),
            step("b", 1),
        ),
    )

    config = {
        "one:v": 2,
        "one:w": 3,
        "split:x:v": 4,
        "split:x:w": 42,
        "choice": "a",
        "choice:a:v": 3,
    }

    expected = Pipeline.create(
        step("one", 1, config={"v": 2, "w": 3}),
        split(
            "split",
            step("x", 1, config={"v": 4, "w": 42}),
            step("y", 1, config=None),
        ),
        step("a", 1, config={"v": 3}),
        name=pipeline.name,
    )

    result = pipeline.configure(config)
    assert result == expected


def test_configuration_with_nested_submodules() -> None:
    pipeline = Pipeline.create(
        step("1", 1, space={"a": [1, 2, 3]}),
        step("2", 2, space={"b": [4, 5, 6]}),
    )

    module1 = Pipeline.create(
        step("3", 3, space={"c": [7, 8, 9]}),
        step("4", 4, space={"d": [10, 11, 12]}),
        name="module1",
    )

    module2 = Pipeline.create(
        choice(
            "choice",
            step("6", 6, space={"e": [13, 14, 15]}),
            step("7", 7, space={"f": [16, 17, 18]}),
        ),
        name="module2",
    )

    module3 = Pipeline.create(
        step("8", 8, space={"g": [19, 20, 21]}),
        name="module3",
    )

    module2 = module2.attach(modules=(module3))

    pipeline = pipeline.attach(modules=(module1, module2))

    config = {
        "1:a": 1,
        "2:b": 4,
        "module1:3:c": 7,
        "module1:4:d": 10,
        "module2:choice": "6",
        "module2:choice:6:e": 13,
        "module2:module3:8:g": 19,
    }

    expected_module1 = Pipeline.create(
        step("3", 3, config={"c": 7}),
        step("4", 4, config={"d": 10}),
        name="module1",
    )

    expected_module2 = Pipeline.create(
        step("6", 6, config={"e": 13}),
        name="module2",
    )

    expected_module3 = Pipeline.create(
        step("8", 8, config={"g": 19}),
        name="module3",
    )

    expected_pipeline = Pipeline.create(
        step("1", 1, config={"a": 1}),
        step("2", 2, config={"b": 4}),
        name=pipeline.name,
    )

    expected_module2 = expected_module2.attach(modules=(expected_module3))

    expected_pipeline = expected_pipeline.attach(
        modules=(expected_module1, expected_module2),
    )

    assert expected_pipeline == pipeline.configure(config)


def test_heirachical_str_with_searchables() -> None:
    pipeline = Pipeline.create(
        step("1", 1, space={"a": [1, 2, 3]}),
        step("2", 2, space={"b": [4, 5, 6]}),
    )

    extra = searchable("searchables", space={"a": [1, 2, 3], "b": [4, 5, 6]})
    pipeline = pipeline.attach(modules=extra)

    config = {
        "1:a": 1,
        "2:b": 4,
        "searchables:a": 1,
        "searchables:b": 4,
    }

    expected = Pipeline.create(
        step("1", 1, config={"a": 1}),
        step("2", 2, config={"b": 4}),
        name=pipeline.name,
    )
    expected_extra = searchable("searchables", config={"a": 1, "b": 4})

    expected = expected.attach(modules=expected_extra)

    assert expected == pipeline.configure(config)
