from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

DEFAULT_SEED = 0


HERE = Path(__file__)


def walk(path: Path, include: str | None = None) -> Iterator[Path]:
    """Yeilds all files, iterating over directory.

    Args:
        path: The root path to walk from
        include: Include only directories which match this string. Defaults to None

    Yields:
        All file paths that could be found from this walk
    """
    for p in path.iterdir():
        if p.is_dir():
            if include is None or re.match(include, p.name):
                yield from walk(p, include)
        else:
            yield p.resolve()


def is_fixture(path: Path) -> bool:
    """Whether a path is a fixture."""
    return path.name.endswith("fixtures.py")


def as_module(path: Path) -> str:
    """Convert a path to a module as seen from here."""
    root = HERE.parent.parent
    parts = path.relative_to(root).parts
    return ".".join(parts).replace(".py", "")


def fixture_modules() -> list[str]:
    """Get all fixture modules."""
    fixtures_folder = HERE.parent / "fixtures"
    if fixtures_folder.exists():
        return [
            as_module(path)
            for path in walk(fixtures_folder)
            if path.name.endswith(".py")
        ]

    return []


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Run before each test."""
    todos = list(item.iter_markers(name="todo"))
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_configure(config) -> None:
    """Used to register marks."""
    config.addinivalue_line("markers", "todo: Mark test as todo")


@pytest.fixture(scope="session")
def event_loop(_):
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


def pytest_sessionfinish(*_: Any) -> None:
    """Called after tests finished, before returning the exit status to the system."""
    # So Dask does something with logging which causes the logging to become
    # a massive mess after the tests run. This is the hack I could find to fix
    # it. However I'm not sure if this causes long term issues to occur.
    # I don't advise reading through the whole thread, it gets a bit messy
    # and caused conflict. Someone even quotes Linus Torvald.
    # https://github.com/pytest-dev/pytest/issues/5502#issuecomment-647157873
    # https://github.com/pytest-dev/pytest/issues/5502#issuecomment-702374419
    import logging

    loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
    for logger in loggers:
        handlers = getattr(logger, "handlers", [])
        for handler in handlers:
            logger.removeHandler(handler)  # type: ignore


pytest_plugins = fixture_modules()
