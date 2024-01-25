import logging
import warnings
from typing import Any

import mkdocs
import mkdocs.plugins
import mkdocs.structure.pages
import markdown_exec.formatters.python

log = logging.getLogger("mkdocs")


@mkdocs.plugins.event_priority(-50)
def on_startup(**kwargs: Any):
    # We get a load of deprecation warnings from SMAC
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ConvergenceWarning from sklearn
    warnings.filterwarnings("ignore", module="sklearn")

    # There's also one code cell in `scheduling.md` that
    # demonstrates that the scheduler needs to be running to submit a task.
    # This casuses a `log.error` to be emitted, which we don't want.


def on_pre_page(
    page: mkdocs.structure.pages.Page,
    config: Any,
    files: Any,
) -> mkdocs.structure.pages.Page | None:
    # NOTE: mkdocs says they're always normalized to be '/' seperated
    # which means this should work on windows as well.
    if page.file.src_uri == "guides/scheduling.md":
        scheduling_logger = logging.getLogger("amltk.scheduling.task")
        scheduling_logger.setLevel(logging.CRITICAL)

    logging.getLogger("smac").setLevel(logging.ERROR)
    logging.getLogger("openml").setLevel(logging.ERROR)
    return page
