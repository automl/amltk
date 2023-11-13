"""A mixin class for allowing rich output."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import closing
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import RenderableType


class RichRenderable(ABC):
    """Mixin for adding rich methods to a class."""

    def _repr_html_(self) -> str:
        """Return an HTML representation of the object."""
        return self._repr_pretty_()

    def _repr_pretty_(self, *_: Any, **__: Any) -> str:
        """Representation for rich printing."""
        from io import StringIO

        import rich

        with closing(StringIO()) as buffer:
            rich.print(self.__rich__(), file=buffer)
            return buffer.getvalue()

    @abstractmethod
    def __rich__(self) -> RenderableType:
        """Return a rich Text object."""
