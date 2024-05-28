"""This module is a hook that when any code is being rendered, it will
print the path to the file being rendered.

This makes it easier to identify which file is being rendered when an error happens."""
import logging
from typing import Any

import mkdocs
import mkdocs.plugins
import mkdocs.structure.pages

log = logging.getLogger("mkdocs")

def on_pre_page(
    page: mkdocs.structure.pages.Page,
    config: Any,
    files: Any,
) -> mkdocs.structure.pages.Page | None:
    log.info(f"{page.file.src_path}")
