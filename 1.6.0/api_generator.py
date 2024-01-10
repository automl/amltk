"""Generate the code reference pages and navigation.

# https://mkdocstrings.github.io/recipes/
"""
from __future__ import annotations

import logging
from pathlib import Path

import mkdocs_gen_files

logger = logging.getLogger(__name__)

for path in sorted(Path("src").rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] in ("__main__", "__version__", "__init__"):
        continue

    if any(part.startswith("_") for part in parts):
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
