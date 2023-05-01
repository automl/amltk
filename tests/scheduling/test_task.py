from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterator

from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Scheduler, Task


@case(tags=["executor"])
def case_thread_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


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

    task = Task(big_memory_function, scheduler, memory_limit=one_half_gb)

    @scheduler.on_start
    def start_task() -> None:
        task(mem_in_bytes=two_gb)

    end_status = scheduler.run(raises=False)

    assert isinstance(end_status, Task.MemoryLimitException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        Task.MEMORY_LIMIT_REACHED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "big_memory_function"): 1,
        (Task.DONE, "big_memory_function"): 1,
        (Task.EXCEPTION, "big_memory_function"): 1,
        (Task.F_EXCEPTION, "big_memory_function"): 1,
        (Task.MEMORY_LIMIT_REACHED, "big_memory_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_time_limited_task(scheduler: Scheduler) -> None:
    task = Task(time_wasting_function, scheduler, wall_time_limit=1)

    @scheduler.on_start
    def start_task() -> None:
        task(duration=3)

    end_status = scheduler.run(raises=False)

    assert isinstance(end_status, Task.WallTimeoutException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        Task.TIMEOUT: 1,
        Task.WALL_TIME_LIMIT_REACHED: 1,
    }

    counts = {
        (Task.SUBMITTED, "time_wasting_function"): 1,
        (Task.DONE, "time_wasting_function"): 1,
        (Task.TIMEOUT, "time_wasting_function"): 1,
        (Task.EXCEPTION, "time_wasting_function"): 1,
        (Task.F_EXCEPTION, "time_wasting_function"): 1,
        (Task.WALL_TIME_LIMIT_REACHED, "time_wasting_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
    assert scheduler.counts == counts


def test_cpu_time_limited_task(scheduler: Scheduler) -> None:
    task = Task(cpu_time_wasting_function, scheduler, cpu_time_limit=1)

    @scheduler.on_start
    def start_task() -> None:
        task(iterations=int(1e16))

    end_status = scheduler.run(raises=False)
    assert isinstance(end_status, Task.CpuTimeoutException)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.EXCEPTION: 1,
        Task.F_EXCEPTION: 1,
        Task.TIMEOUT: 1,
        Task.CPU_TIME_LIMIT_REACHED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "cpu_time_wasting_function"): 1,
        (Task.DONE, "cpu_time_wasting_function"): 1,
        (Task.EXCEPTION, "cpu_time_wasting_function"): 1,
        (Task.F_EXCEPTION, "cpu_time_wasting_function"): 1,
        (Task.TIMEOUT, "cpu_time_wasting_function"): 1,
        (Task.CPU_TIME_LIMIT_REACHED, "cpu_time_wasting_function"): 1,
        Scheduler.STARTED: 1,
        Scheduler.STOP: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_concurrency_limit_of_tasks(scheduler: Scheduler) -> None:
    task = Task(time_wasting_function, scheduler, concurrent_limit=2)

    @scheduler.on_start(repeat=10)
    def launch_many() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.counts == {
        Task.CONCURRENT_LIMIT_REACHED: 8,
        Task.SUBMITTED: 2,
        Task.DONE: 2,
        Task.RETURNED: 2,
        Task.F_RETURNED: 2,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "time_wasting_function"): 2,
        (Task.CONCURRENT_LIMIT_REACHED, "time_wasting_function"): 8,
        (Task.DONE, "time_wasting_function"): 2,
        (Task.RETURNED, "time_wasting_function"): 2,
        (Task.F_RETURNED, "time_wasting_function"): 2,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_call_limit_of_tasks(scheduler: Scheduler) -> None:
    task = Task(time_wasting_function, scheduler, call_limit=2)

    @scheduler.on_start(repeat=10)
    def launch() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.counts == {
        Task.CALL_LIMIT_REACHED: 8,
        Task.SUBMITTED: 2,
        Task.DONE: 2,
        Task.RETURNED: 2,
        Task.F_RETURNED: 2,
    }

    assert scheduler.counts == {
        (Task.CALL_LIMIT_REACHED, "time_wasting_function"): 8,
        (Task.SUBMITTED, "time_wasting_function"): 2,
        (Task.DONE, "time_wasting_function"): 2,
        (Task.RETURNED, "time_wasting_function"): 2,
        (Task.F_RETURNED, "time_wasting_function"): 2,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
