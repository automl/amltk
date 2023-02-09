from byop import Pipeline, choice, split, step


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
            step("x", 1, config=None),
            step("y", 1, space={"v": [4, 5, 6]}),
        ),
        choice(
            "choice",
            step("a", 1, config=None),
            step("b", 1, config=None),
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
