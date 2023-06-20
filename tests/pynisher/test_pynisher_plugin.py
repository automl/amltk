from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Iterator

from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.pynisher import PynisherPlugin
from amltk.scheduling import Scheduler, Task


@case(tags=["executor"])
def case_process_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_dask_executor() -> ClientExecutor:
    # Dask will raise a warning when re-using the ports, hence
    # we silence the warnings here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster = LocalCluster(
            n_workers=2,
            silence_logs=logging.ERROR,
            worker_class=Worker,
            processes=True,
        )

    client = Client(cluster)
    executor = client.get_executor()
    assert isinstance(executor, ClientExecutor)
    return executor


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Iterator[Scheduler]:
    yield Scheduler(executor)


def big_memory_function(mem_in_bytes: int) -> bytearray:
    z = bytearray(mem_in_bytes)
    return z  # noqa: RET504


def time_wasting_function(duration: int) -> int:
    time.sleep(duration)
    return duration


def cpu_time_wasting_function(iterations: int) -> int:
    while iterations > 0:
        iterations -= 1
    return iterations


def test_memory_limited_task(scheduler: Scheduler) -> None:
    one_half_gb = int(1e9 * 1.5)
    two_gb = int(1e9) * 2

    task = Task(
        big_memory_function,
        scheduler,
        plugins=[PynisherPlugin(memory_limit=one_half_gb)],
    )

    @scheduler.on_start
    def start_task() -> None:
        task(mem_in_bytes=two_gb)

    end_status = scheduler.run(end_on_exception=True, raises=False)

    assert isinstance(end_status, PynisherPlugin.MemoryLimitException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.F_SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        PynisherPlugin.MEMORY_LIMIT_REACHED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "big_memory_function"): 1,
        (Task.F_SUBMITTED, "big_memory_function"): 1,
        (Task.DONE, "big_memory_function"): 1,
        (Task.EXCEPTION, "big_memory_function"): 1,
        (Task.F_EXCEPTION, "big_memory_function"): 1,
        (PynisherPlugin.MEMORY_LIMIT_REACHED, "big_memory_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_time_limited_task(scheduler: Scheduler) -> None:
    task = Task(
        time_wasting_function,
        scheduler,
        plugins=[PynisherPlugin(wall_time_limit=1)],
    )

    @scheduler.on_start
    def start_task() -> None:
        task(duration=3)

    end_status = scheduler.run(raises=False)

    assert isinstance(end_status, PynisherPlugin.WallTimeoutException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.F_SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        PynisherPlugin.TIMEOUT: 1,
        PynisherPlugin.WALL_TIME_LIMIT_REACHED: 1,
    }

    counts = {
        (Task.SUBMITTED, "time_wasting_function"): 1,
        (Task.F_SUBMITTED, "time_wasting_function"): 1,
        (Task.DONE, "time_wasting_function"): 1,
        (PynisherPlugin.TIMEOUT, "time_wasting_function"): 1,
        (Task.EXCEPTION, "time_wasting_function"): 1,
        (Task.F_EXCEPTION, "time_wasting_function"): 1,
        (PynisherPlugin.WALL_TIME_LIMIT_REACHED, "time_wasting_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
    assert scheduler.counts == counts


def test_cpu_time_limited_task(scheduler: Scheduler) -> None:
    task = Task(
        cpu_time_wasting_function,
        scheduler,
        plugins=[PynisherPlugin(cpu_time_limit=1)],
    )

    @scheduler.on_start
    def start_task() -> None:
        task(iterations=int(1e16))

    end_status = scheduler.run(raises=False)
    assert isinstance(end_status, PynisherPlugin.CpuTimeoutException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.F_SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        PynisherPlugin.TIMEOUT: 1,
        PynisherPlugin.CPU_TIME_LIMIT_REACHED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "cpu_time_wasting_function"): 1,
        (Task.F_SUBMITTED, "cpu_time_wasting_function"): 1,
        (Task.DONE, "cpu_time_wasting_function"): 1,
        (Task.EXCEPTION, "cpu_time_wasting_function"): 1,
        (Task.F_EXCEPTION, "cpu_time_wasting_function"): 1,
        (PynisherPlugin.TIMEOUT, "cpu_time_wasting_function"): 1,
        (PynisherPlugin.CPU_TIME_LIMIT_REACHED, "cpu_time_wasting_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
