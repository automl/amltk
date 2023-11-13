from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rich.console import RenderableType

DEFAULT_MKDOCS_CODE_BLOCK_WIDTH = 80


SKLEARN_LINK = "https://www.scikit-learn.org/stable/modules/generated/{0}.html"


def sklearn_link_generator(name: str) -> str:
    """Generate a link for a sklearn function."""
    reduced_name = ".".join(s for s in name.split(".") if not s.startswith("_"))
    return SKLEARN_LINK.format(reduced_name)


@lru_cache
def _try_get_link(fully_scoped_name: str) -> str | None:
    """Try to get a link for a string.

    Expects fully qualified import names.
    """
    from amltk.options import _amltk_options

    links = _amltk_options.get("links", {})

    for k, v in links.items():
        if fully_scoped_name.startswith(k):
            if isinstance(v, str):
                return v
            if callable(v):
                return v(fully_scoped_name)

    return None


def link(obj: Any) -> str | None:
    """Try to get a link for an object."""
    from amltk._functional import fullname

    return _try_get_link(fullname(obj))


def make_picklable(thing: Any, name: str | None = None) -> None:
    """This is hack to make the examples code with schedulers work.

    Scheduler uses multiprocessing and multiprocessing requires that
    all objects passed to the scheduler are picklable. This is not
    the case for the classes/functions defined in the example code.
    """
    import __main__

    _name = thing.__name__ if name is None else name
    setattr(__main__, _name, thing)


def as_rich_svg(
    *renderable: RenderableType,
    title: str = "",
    width: int = DEFAULT_MKDOCS_CODE_BLOCK_WIDTH,
) -> str:
    from rich.console import Console

    with open(os.devnull, "w") as devnull:  # noqa: PTH123
        console = Console(record=True, width=width, file=devnull, markup=False)
        for r in renderable:
            console.print(r, markup=False)

    return console.export_svg(title=title)


HTML_FORMAT = """
<pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;font-size:{fontsize}">
<code style="font-family:inherit">{{code}}</code>
</pre>
"""  # noqa: E501

SIZES = {
    "very-small": "0.5rem",
    "small": "0.7rem",
    "medium": "0.75rem",
    "large": "1rem",
}


def as_rich_html(
    *renderable: RenderableType,
    width: int = DEFAULT_MKDOCS_CODE_BLOCK_WIDTH,
    fontsize: str = "medium",
) -> str:
    from rich.console import Console

    with open(os.devnull, "w") as devnull:  # noqa: PTH123
        console = Console(record=True, width=width, file=devnull)
        for r in renderable:
            console.print(r, markup=False, highlight=False)

        fontsize = SIZES.get(fontsize, fontsize)
        code_format = HTML_FORMAT.format(fontsize=fontsize)
        return console.export_html(inline_styles=True, code_format=code_format)


def doc_print(
    _print: Callable[[str], Any],
    *renderable: RenderableType,
    title: str = "",
    width: int = DEFAULT_MKDOCS_CODE_BLOCK_WIDTH,
    output: Literal["svg", "html"] = "html",
    fontsize: str = "medium",
) -> None:
    if output == "svg":
        _print(as_rich_svg(*renderable, title=title, width=width))
    elif len(renderable) == 1:
        try:
            from sklearn.base import BaseEstimator, TransformerMixin
            from sklearn.pipeline import Pipeline

            if isinstance(
                renderable[0],
                Pipeline | BaseEstimator() | TransformerMixin(),
            ):
                _print(renderable[0]._repr_html_())  # type: ignore
                return
        except Exception:  # noqa: BLE001
            _print(as_rich_html(*renderable, width=width, fontsize=fontsize))
    else:
        _print(as_rich_html(*renderable, width=width, fontsize=fontsize))
