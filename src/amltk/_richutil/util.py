"""Utilities for working with rich."""
# NOTE: All rich imports should be done locally to prevent any issue
# where rich not being installed.
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

from amltk._richutil.renderers import (
    ProcessPoolExecutorRenderer,
    rich_make_column_selector,
)
from amltk.types import safe_isinstance

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import OverflowMethod, RenderableType
    from rich.pretty import Pretty
    from rich.style import StyleType
    from rich.table import Table
    from rich.text import TextType


def richify(
    obj: Any,
    *,
    otherwise: type[Pretty] | None = None,
) -> Any | RenderableType:
    """Try to convert an object to a rich object.

    This is mainly for common objects we may encounter and
    want to prettify if we can.

    Args:
        obj: Object to convert.
        otherwise: If not None, return this called with obj.
            Else, return obj.
    """
    if safe_isinstance(obj, "make_column_selector"):
        return rich_make_column_selector(obj)

    if isinstance(obj, ProcessPoolExecutor):
        return ProcessPoolExecutorRenderer(obj)

    if otherwise:
        return otherwise(obj)

    return obj


def df_to_table(
    df: pd.DataFrame,
    *,
    title: TextType | None = None,
    index_style: StyleType = "",
    expand: bool = True,
    overflow: OverflowMethod = "ellipsis",
) -> Table:
    """Convert a dataframe to a rich table."""
    from rich.table import Column, Table

    index_name = df.index.name if df.index.name is not None else "Index"
    headers = [
        Column(str(index_name), justify="left", style=index_style, overflow=overflow),
    ]
    headers.extend(df.columns)

    table = Table(*headers, title=title, title_justify="left", expand=expand)
    for index, row in df.iterrows():
        table.add_row(str(index), *[str(cell) for cell in row])

    return table


def is_jupyter() -> bool:
    """Return True if running in a Jupyter environment."""
    # https://github.com/Textualize/rich/blob/fd981823644ccf50d685ac9c0cfe8e1e56c9dd35/rich/console.py#L518-L535
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: B018
    except NameError:
        return False
    ipython = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    shell = ipython.__class__.__name__
    if (
        "google.colab" in str(ipython.__class__)
        or os.getenv("DATABRICKS_RUNTIME_VERSION")
        or shell == "ZMQInteractiveShell"
    ):
        return True  # Jupyter notebook or qtconsole

    if shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython

    return False  # Other type (?)
