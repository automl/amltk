from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time
from typing import Iterator
import warnings

from dask.distributed import Client, LocalCluster
from distributed.cfexecutor import ClientExecutor
from pynisher import CpuTimeoutException, MemoryLimitException, WallTimeoutException
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Scheduler, SchedulerEvent, TaskEvent


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
        cluster = LocalCluster(n_workers=2, silence_logs=logging.ERROR, processes=False)
    client = Client(cluster)
    return client.get_executor()


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

    task = scheduler.task(big_memory_function, memory_limit=one_half_gb)
    scheduler.on_start(lambda: task(mem_in_bytes=two_gb))

    errors: list[BaseException] = []
    task.on_exception(errors.append)

    end_status = scheduler.run(end_on_empty=True)

    assert end_status == scheduler.exitcodes.EXHAUSTED

    assert len(errors) == 1
    assert isinstance(errors[0], MemoryLimitException)

    counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.DONE: 1,
        TaskEvent.EXCEPTION: 1,
        ("big_memory_function", TaskEvent.SUBMITTED): 1,
        ("big_memory_function", TaskEvent.DONE): 1,
        ("big_memory_function", TaskEvent.EXCEPTION): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
    }
    assert dict(scheduler.counts) == counts


def test_time_limited_task(scheduler: Scheduler) -> None:
    task = scheduler.task(time_wasting_function, wall_time_limit=1)

    scheduler.on_start(lambda: task(duration=3))

    errors: list[BaseException] = []
    task.on_exception(errors.append)

    end_status = scheduler.run(end_on_empty=True)

    assert end_status == scheduler.exitcodes.EXHAUSTED

    assert len(errors) == 1
    assert isinstance(errors[0], WallTimeoutException)

    counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.DONE: 1,
        TaskEvent.EXCEPTION: 1,
        ("time_wasting_function", TaskEvent.SUBMITTED): 1,
        ("time_wasting_function", TaskEvent.DONE): 1,
        ("time_wasting_function", TaskEvent.EXCEPTION): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
    }
    assert dict(scheduler.counts) == counts


def test_cpu_time_limited_task(scheduler: Scheduler) -> None:
    task = scheduler.task(cpu_time_wasting_function, cpu_time_limit=1)

    scheduler.on_start(lambda: task(iterations=int(1e16)))

    errors: list[BaseException] = []
    task.on_exception(errors.append)

    end_status = scheduler.run(end_on_empty=True)

    assert end_status == scheduler.exitcodes.EXHAUSTED

    assert len(errors) == 1
    assert isinstance(errors[0], CpuTimeoutException)

    counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.DONE: 1,
        TaskEvent.EXCEPTION: 1,
        ("cpu_time_wasting_function", TaskEvent.SUBMITTED): 1,
        ("cpu_time_wasting_function", TaskEvent.DONE): 1,
        ("cpu_time_wasting_function", TaskEvent.EXCEPTION): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
    }
    assert dict(scheduler.counts) == counts


def test_concurrency_limit_of_tasks(scheduler: Scheduler) -> None:
    task = scheduler.task(time_wasting_function, concurrent_limit=2)

    results: list[int] = []
    task.on_return(results.append)

    def launch_many() -> None:
        for _ in range(10):
            task(duration=2)

    scheduler.on_start(launch_many)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == scheduler.exitcodes.EXHAUSTED
    assert len(results) == 2

    counts = {
        TaskEvent.SUBMITTED: 2,
        TaskEvent.DONE: 2,
        TaskEvent.RETURNED: 2,
        ("time_wasting_function", TaskEvent.SUBMITTED): 2,
        ("time_wasting_function", TaskEvent.DONE): 2,
        ("time_wasting_function", TaskEvent.RETURNED): 2,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
    }
    assert dict(scheduler.counts) == counts
