"""This module is concerned with the termination of the workers of a scheduler.

Most Executors on `shutdown(wait=False)` will not terminate their workers
but just cancel pending futures. This means that currently running tasks
will continue to run until they finish and have their result discarded as
the program will have moved on.

We provide some custom strategies for known executors.

Note:
    There is no known way in basic Python to forcibully terminate a thread
    that does not account for early terminiation explicitly.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from typing import TypeVar

import psutil

_Executor = TypeVar("_Executor", bound=Executor)


def _terminate_with_psutil(executor: ProcessPoolExecutor) -> None:
    """Terminate all processes in the given executor using psutil.

    Should work for ProcessPoolExecutor cross platform.

    Args:
        executor: The executor to terminate.
    """
    # They've already finished
    if not executor._processes:
        return

    worker_processes = [psutil.Process(p.pid) for p in executor._processes.values()]
    for worker_process in worker_processes:
        try:
            child_preocesses = worker_process.children(recursive=True)
        except psutil.NoSuchProcess:
            continue

        for child_process in child_preocesses:
            with suppress(psutil.NoSuchProcess):
                child_process.terminate()

        with suppress(psutil.NoSuchProcess):
            worker_process.terminate()


def termination_strategy(executor: _Executor) -> Callable[[_Executor], None] | None:
    """Return a termination strategy for the given executor.

    Args:
        executor: The executor to get a termination strategy for.

    Returns:
        A termination strategy for the given executor, or None if no
        termination strategy is available.
    """
    if isinstance(executor, ThreadPoolExecutor):
        return None

    if isinstance(executor, ProcessPoolExecutor):
        return _terminate_with_psutil  # type: ignore

    # Dask process based things seem pretty happy to close nicely and need
    # no special treatment.

    return None
