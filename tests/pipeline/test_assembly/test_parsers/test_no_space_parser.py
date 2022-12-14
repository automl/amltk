from byop.pipeline import Pipeline, step
from byop.pipeline.api import split
from byop.spaces.parsers import NoSpaceParser


def test_none_parser_on_blank_pipeline() -> None:
    pipeline = Pipeline.create(
        step("a", 1),
        step("b", 2),
        split("a", step("c", 3)),
    )

    result = NoSpaceParser.parse(pipeline)
    assert result.is_ok()


def test_none_fails_pipeline_with_space() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"alpha": [1, 2, 3]}),
        step("b", 2),
        split("a", step("c", 3)),
    )

    result = NoSpaceParser.parse(pipeline)
    assert result.is_err()
