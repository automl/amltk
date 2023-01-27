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
