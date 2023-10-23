"""Render a function/class signature."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from amltk.links import link
from amltk.options import _amltk_options

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.highlighter import Highlighter
    from rich.style import StyleType


@dataclass
class Function:
    """Render a function/class signature."""

    f: Callable | type
    signature: bool = field(
        default_factory=lambda: _amltk_options.get("rich_signatures", True),
    )
    link: Literal["auto"] | str | Literal[False] = field(
        default_factory=lambda: _amltk_options.get("rich_link", "auto"),
    )
    highlighter: Highlighter | None = None

    def __rich__(self) -> RenderableType:
        # Taken from rich._inspect._get_signature
        from rich.highlighter import ReprHighlighter
        from rich.style import Style
        from rich.text import Text

        f = self.f

        if self.signature:
            try:
                _signature = str(inspect.signature(f))
            except ValueError:
                _signature = "(...)"
        else:
            _signature = "()"

        highlighter = self.highlighter or ReprHighlighter()
        signature_text = highlighter(_signature)

        qualname = getattr(f, "__qualname__", "<function>")

        _name_style: StyleType = "inspect.callable"
        _link = None
        if self.link == "auto":
            _link = link(f)
            if _link is not None:
                _name_style = Style(link=_link, underline=True)
        elif isinstance(self.link, str):
            _link = self.link
            _name_style = Style(link=_link, underline=True)

        # If obj is a module, there may be classes (which are callable) to display
        if inspect.isclass(f):
            prefix = "class"
        elif inspect.iscoroutinefunction(f):
            prefix = "async def"
        else:
            prefix = "def"

        return Text.assemble(
            (f"{prefix} ", f"inspect.{prefix.replace(' ', '_')}"),
            (qualname, _name_style),
            signature_text,
        )


Class = Function
"""Alias for Function."""
