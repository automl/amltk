from __future__ import annotations

from pytest_cases import parametrize

from byop.pipeline import Parser, Pipeline, split, step


@parametrize("parser", Parser.default_parsers())
def test_none_parser_on_blank_pipeline(parser: Parser) -> None:
    pipeline = Pipeline.create(
        step("a", 1),
        step("b", 2),
        split("a", step("c", 3)),
    )

    result = pipeline.space(parser=parser)
    assert result == parser.empty()


@parametrize("parser", Parser.default_parsers())
def test_none_fails_pipeline_with_space(parser: Parser) -> None:
    pipeline = Pipeline.create(
        step("a", 1, space={"alpha": [1, 2, 3]}),
        step("b", 2),
        split("a", step("c", 3)),
    )

    result = pipeline.space(parser=parser)
    assert result != parser.empty()
