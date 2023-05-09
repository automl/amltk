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
from byop.scheduling.task_plugin import CallLimiter


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


def test_concurrency_limit_of_tasks(scheduler: Scheduler) -> None:
    task = Task(
        time_wasting_function,
        scheduler=scheduler,
        plugins=[CallLimiter(max_concurrent=2)],
    )

    @scheduler.on_start(repeat=10)
    def launch_many() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.counts == {
        CallLimiter.CONCURRENT_LIMIT_REACHED: 8,
        Task.SUBMITTED: 2,
        Task.DONE: 2,
        Task.RETURNED: 2,
        Task.F_RETURNED: 2,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "time_wasting_function"): 2,
        (CallLimiter.CONCURRENT_LIMIT_REACHED, "time_wasting_function"): 8,
        (Task.DONE, "time_wasting_function"): 2,
        (Task.RETURNED, "time_wasting_function"): 2,
        (Task.F_RETURNED, "time_wasting_function"): 2,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_call_limit_of_tasks(scheduler: Scheduler) -> None:
    task = Task(
        time_wasting_function,
        scheduler,
        plugins=[CallLimiter(max_calls=2)],
    )

    @scheduler.on_start(repeat=10)
    def launch() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.counts == {
        CallLimiter.CALL_LIMIT_REACHED: 8,
        Task.SUBMITTED: 2,
        Task.DONE: 2,
        Task.RETURNED: 2,
        Task.F_RETURNED: 2,
    }

    assert scheduler.counts == {
        (CallLimiter.CALL_LIMIT_REACHED, "time_wasting_function"): 8,
        (Task.SUBMITTED, "time_wasting_function"): 2,
        (Task.DONE, "time_wasting_function"): 2,
        (Task.RETURNED, "time_wasting_function"): 2,
        (Task.F_RETURNED, "time_wasting_function"): 2,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
