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


def polite_kill(process: psutil.Process, timeout: int | None = None) -> None:
    """Politely kill a process.

    This works by first sending a SIGTERM to the process, and then if it
    doesn't respond to that, sending a SIGKILL.

    On Windows, SIGTERM is not available, so `terminate()` will
    send a `SIGKILL` directly.

    Args:
        process: The process to kill.
        timeout: The time to wait for the process after sending SIGTERM.
            before resorting to SIGKILL. If None, wait indefinitely.
    """
    with suppress(psutil.NoSuchProcess):
        process.terminate()
        process.wait(timeout=timeout)

        # Forcibly kill it if it's not responding to the SIGTERM
        if process.is_running():
            process.kill()


def _terminate_with_psutil(executor: ProcessPoolExecutor) -> None:
    """Terminate all processes in the given executor using psutil.

    Should work for ProcessPoolExecutor cross platform.

    Args:
        executor: The executor to terminate.
    """
    # They've already finished
    if not executor._processes:
        return

    for process in executor._processes.values():
        try:
            worker_process = psutil.Process(process.pid)
            # We reverse here to start from leaf processes first, giving parents
            # time to cleanup after their terminated subprocesses.
            child_processes = reversed(worker_process.children(recursive=True))
        except psutil.NoSuchProcess:
            continue

        for child_process in child_processes:
            polite_kill(child_process, timeout=5)

        polite_kill(worker_process, timeout=5)


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
