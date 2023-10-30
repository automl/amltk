from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from rich.console import RenderableType

DEFAULT_MKDOCS_CODE_BLOCK_WIDTH = 82
FONTSIZE_RICH_HTML = "0.5rem"


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
    "small": "0.75rem",
    "medium": "0.8rem",
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
    else:
        _print(as_rich_html(*renderable, width=width, fontsize=fontsize))
