"""Options for the AMTLK package.

In general, these options are not intended for functional differences but
more for any output generated by the package, such as rich or logging.
"""
from __future__ import annotations

from typing import Callable, Literal, TypedDict

from amltk.links import sklearn_link_generator


class AMLTKOptions(TypedDict):
    """The options available for AMTLK.

    ```python exec="true" source="material-block" result="python"
    from amltk import options

    print(options)
    ```
    """

    rich_signatures: bool
    """Whether to display full signatures in rich output."""

    rich_link: Literal["auto", False]
    """Whether to use links in rich output."""

    links: dict[str, str | Callable[[str], str]]
    """The links to use in rich output.

    The keys are the names of the packages, and the values are either the

    """


_amltk_options: AMLTKOptions = {
    "rich_signatures": True,
    "rich_link": "auto",
    "links": {"sklearn": sklearn_link_generator},
}
