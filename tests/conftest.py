# pylint: disable=unused-argument
from typing import Iterator, List, Optional

import re
from pathlib import Path

import pytest
from pytest import Item

DEFAULT_SEED = 0


HERE = Path(__file__)


def walk(path: Path, include: Optional[str] = None) -> Iterator[Path]:
    """Yeilds all files, iterating over directory
    Parameters
    ----------
    path: Path
        The root path to walk from
    include: Optional[str] = None
        Include only directories which match this string
    Returns
    -------
    Iterator[Path]
        All file paths that could be found from this walk
    """
    for p in path.iterdir():
        if p.is_dir():
            if include is None or re.match(include, p.name):
                yield from walk(p, include)
        else:
            yield p.resolve()


def is_fixture(path: Path) -> bool:
    """Whether a path is a fixture"""
    return path.name.endswith("fixtures.py")


def as_module(path: Path) -> str:
    """Convert a path to a module as seen from here"""
    root = HERE.parent.parent
    parts = path.relative_to(root).parts
    return ".".join(parts).replace(".py", "")


def fixture_modules() -> List[str]:
    """Get all fixture modules"""
    fixtures_folder = HERE.parent / "fixtures"
    return [
        as_module(path) for path in walk(fixtures_folder) if path.name.endswith(".py")
    ]


def pytest_runtest_setup(item: Item) -> None:
    """Run before each test"""
    todos = list(item.iter_markers(name="todo"))
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_configure(config) -> None:
    """Used to register marks"""
    config.addinivalue_line("markers", "todo: Mark test as todo")


pytest_plugins = fixture_modules()
