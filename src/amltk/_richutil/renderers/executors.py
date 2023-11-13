"""Renderers for known Executors."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

from amltk._richutil.renderable import RichRenderable

if TYPE_CHECKING:
    from rich.panel import Panel


@dataclass
class ProcessPoolExecutorRenderer(RichRenderable):
    """Render a ProcessPoolExecutor."""

    executor: ProcessPoolExecutor

    @override
    def __rich__(self) -> Panel:
        from rich.panel import Panel
        from rich.pretty import Pretty
        from rich.text import Text

        max_workers = self.executor._max_workers  # type: ignore
        return Panel.fit(
            Pretty({"max_workers": max_workers}),
            title=Text("ProcessPoolExecutor", no_wrap=True),
            title_align="left",
        )
