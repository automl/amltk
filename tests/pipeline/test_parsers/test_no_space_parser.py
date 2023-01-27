import pytest

from byop import ParseError, Pipeline, split, step
from byop.parsing import NoSpaceParser


def test_none_parser_on_blank_pipeline() -> None:
    pipeline = Pipeline.create(
        step("a", 1),
        step("b", 2),
        split("a", step("c", 3)),
    )

    result = pipeline.space(parser="auto")
    assert result is None


def test_none_fails_pipeline_with_space() -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"alpha": [1, 2, 3]}),
        step("b", 2),
        split("a", step("c", 3)),
    )

    with pytest.raises(ParseError):
        pipeline.space(parser=NoSpaceParser)
