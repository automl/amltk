"""Render a function/class signature."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

from amltk._doc import link
from amltk._functional import funcname
from amltk._richutil.renderable import RichRenderable
from amltk.options import _amltk_options

if TYPE_CHECKING:
    from rich.highlighter import Highlighter
    from rich.style import StyleType
    from rich.text import Text


@dataclass
class Function(RichRenderable):
    """Render a function/class signature."""

    f: Callable | type
    signature: bool | Literal["..."] | tuple[tuple, dict] = field(
        default_factory=lambda: _amltk_options.get("rich_signatures", True),
    )
    link: Literal["auto"] | str | Literal[False] = field(
        default_factory=lambda: _amltk_options.get("rich_link", "auto"),
    )
    highlighter: Highlighter | None = None
    prefix: str | None = None
    no_wrap: bool = False

    @override
    def __rich__(self) -> Text:  # noqa: C901, PLR0912, PLR0915
        # Taken from rich._inspect._get_signature
        from rich.highlighter import ReprHighlighter
        from rich.style import Style
        from rich.text import Text

        f = self.f
        qualname = getattr(f, "__qualname__", None)
        if qualname is None:
            qualname = funcname(f, default="<function>")

        is_lambda = False
        if "<lambda>" in qualname:
            qualname = qualname.replace("<lambda>", "λ")
            is_lambda = True

        if self.signature is True:
            try:
                _signature = str(inspect.signature(f))
            except ValueError:
                _signature = "(...)"
        elif self.signature == "...":
            _signature = "(...)"
        elif isinstance(self.signature, tuple):
            _signature = "("
            args, kwargs = self.signature
            if args:
                _signature += ", ".join(map(str, args))
            if kwargs:
                if args:
                    _signature += ", "
                _signature += ", ".join(
                    f"{key}={value}" for key, value in kwargs.items()
                )
            _signature += ")"
        else:
            _signature = "()"

        highlighter = self.highlighter or ReprHighlighter()
        signature_text = highlighter(_signature)

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
        if self.prefix is not None:
            prefix, style = self.prefix, "inspect.class"
        elif is_lambda:
            prefix, style = "lambda", "yellow italic"
        elif inspect.isclass(f):
            prefix, style = "class", "inspect.class"
        elif inspect.iscoroutinefunction(f):
            prefix, style = "async def", "inspect.async_def"

        else:
            prefix, style = "def", "inspect.def"

        return Text.assemble(
            (f"{prefix} ", style),
            (qualname, _name_style),
            signature_text,
            no_wrap=self.no_wrap,
        )


Class = Function
"""Alias for Function."""
