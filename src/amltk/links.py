"""Links to documentation pages."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

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
    from amltk.functional import fullname

    return _try_get_link(fullname(obj))
