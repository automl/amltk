from __future__ import annotations

import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parent.resolve()
EXAMPLE_DIR = HERE.parent / "examples"
example_files = EXAMPLE_DIR.rglob("*.py")


def is_runnable(example: Path) -> bool:
    with example.open() as f:
        # If `doc-runnable` is not in the first 10 lines,
        # we don't consider it testable.
        for _ in range(10):
            line = next(f)
            if "doc-runnable" in line.lower():
                return True
        return False


parameters = [
    pytest.param(example, id=str(example.relative_to(EXAMPLE_DIR)))
    for example in example_files
    if is_runnable(example)
]


@pytest.mark.parametrize("example", parameters)
@pytest.mark.example()
def test_example(example: Path) -> None:
    runpy.run_path(str(example), run_name="__main__")
