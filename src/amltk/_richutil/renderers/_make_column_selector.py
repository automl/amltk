"""Defines custom renderers for objects to be displayed nicer with rich."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

if TYPE_CHECKING:
    from sklearn.compose import make_column_selector


class rich_make_column_selector:  # noqa: N801
    """Defines better `repr` for make_column_selector when displayed with rich."""

    def __init__(self, col_selector: make_column_selector) -> None:
        """Initialize."""
        super().__init__()
        self.col_selector = col_selector

    @override
    def __repr__(self) -> str:
        _str = ", ".join(
            f"{k}={v}" for k, v in self.col_selector.__dict__.items() if v is not None
        )
        return f"make_column_selector({_str})"
