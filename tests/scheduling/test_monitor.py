from __future__ import annotations

import pandas as pd

from amltk.scheduling import Scheduler
from amltk.scheduling.queue_monitor import QueueMonitor


def fast_f(x: int) -> int:
    return x + 1


def test_queue_monitor() -> None:
    N_WORKERS = 2
    TIMEOUT = 1
    TIMEOUT_NS = TIMEOUT * 1e9
    scheduler = Scheduler.with_processes(max_workers=N_WORKERS)
    monitor = QueueMonitor(scheduler)
    task = scheduler.task(fast_f)

    @scheduler.on_start(repeat=N_WORKERS)
    def start():
        task.submit(1)

    @task.on_result
    def result(_, x: int):
        if scheduler.running():
            task.submit(x)

    scheduler.run(timeout=TIMEOUT, wait=False)
    df = monitor.df()

    # Queue size should always be less than or equal to the number of workers.
    assert (df["queue_size"].max() <= N_WORKERS).all()
    assert (df["queue_size"] + df["idle"] == N_WORKERS).all()

    pd.testing.assert_series_equal(
        df["queued"] + df["finished"] + df["cancelled"],
        df["queue_size"],
        check_names=False,
    )
    assert df.index.is_monotonic_increasing

    # If we specify that is has more workers, it should be reflected in the queue size + idle.
    df = monitor.df(n_workers=N_WORKERS + 1)
    assert (df["queue_size"] + df["idle"] == N_WORKERS + 1).all()
