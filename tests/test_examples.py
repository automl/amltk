from __future__ import annotations

import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parent.resolve()
EXAMPLE_DIR = HERE.parent / "examples"
example_files = EXAMPLE_DIR.rglob("*.py")

parameters = [
    pytest.param(example, id=str(example.relative_to(EXAMPLE_DIR)))
    for example in example_files
]


@pytest.mark.parametrize("example", parameters)
@pytest.mark.example()
def test_example(example: Path) -> None:
    runpy.run_path(str(example), run_name="__main__")
