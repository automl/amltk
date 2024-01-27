from __future__ import annotations

import logging
import time
import warnings
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor

import pytest
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.optimization import Trial
from amltk.scheduling import ExitState, Scheduler
from amltk.scheduling.plugins.pynisher import PynisherPlugin


@case(tags=["executor"])
def case_process_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_loky_executor() -> ProcessPoolExecutor:
    from loky import get_reusable_executor

    return get_reusable_executor(max_workers=2)  # type: ignore


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


@fixture(scope="function")
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Iterator[Scheduler]:
    yield Scheduler(executor)


def big_memory_function(mem_in_bytes: int) -> bytearray:
    z = bytearray(mem_in_bytes)
    return z  # noqa: RET504


def trial_with_big_memory(trial: Trial, mem_in_bytes: int) -> Trial.Report:
    big_memory_function(mem_in_bytes)
    return trial.success()


def trial_with_time_wasting(trial: Trial, duration: int) -> Trial.Report:
    time_wasting_function(duration)
    return trial.success()


def time_wasting_function(duration: int) -> int:
    time.sleep(duration)
    return duration


def cpu_time_wasting_function(iterations: int) -> int:
    while iterations > 0:
        iterations -= 1
    return iterations


def test_memory_limited_task(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("memory"):
        pytest.skip("Pynisher does not support memory limits on this system")

    one_half_gb = int(1e9 * 1.5)
    two_gb = int(1e9) * 2

    pynisher = PynisherPlugin(memory_limit=one_half_gb)
    task = scheduler.task(big_memory_function, plugins=pynisher)

    @scheduler.on_start
    def start_task() -> None:
        task.submit(mem_in_bytes=two_gb)

    with pytest.raises(PynisherPlugin.MemoryLimitException):
        scheduler.run(on_exception="raise")

    assert task.event_counts == Counter(
        {
            task.SUBMITTED: 1,
            task.DONE: 1,
            task.EXCEPTION: 1,
            pynisher.MEMORY_LIMIT_REACHED: 1,
        },
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.STOP: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
            scheduler.FUTURE_EXCEPTION: 1,
        },
    )


def test_time_limited_task(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("wall_time"):
        pytest.skip("Pynisher does not support wall time limits on this system")

    task = scheduler.task(
        time_wasting_function,
        plugins=PynisherPlugin(walltime_limit=1),
    )

    @scheduler.on_start
    def start_task() -> None:
        task.submit(duration=3)

    with pytest.raises(PynisherPlugin.WallTimeoutException):
        scheduler.run(on_exception="raise")

    assert task.event_counts == Counter(
        {
            task.SUBMITTED: 1,
            task.DONE: 1,
            task.EXCEPTION: 1,
            PynisherPlugin.TIMEOUT: 1,
            PynisherPlugin.WALL_TIME_LIMIT_REACHED: 1,
        },
    )

    counts = Counter(
        {
            scheduler.STARTED: 1,
            scheduler.STOP: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
            scheduler.FUTURE_EXCEPTION: 1,
        },
    )
    assert scheduler.event_counts == counts


def test_cpu_time_limited_task(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("cpu_time"):
        pytest.skip("Pynisher does not support cpu time limits on this system")

    task = scheduler.task(
        cpu_time_wasting_function,
        plugins=PynisherPlugin(cputime_limit=1),
    )

    @scheduler.on_start
    def start_task() -> None:
        task.submit(iterations=int(1e16))

    with pytest.raises(PynisherPlugin.CpuTimeoutException):
        scheduler.run(on_exception="raise")

    assert task.event_counts == Counter(
        {
            task.SUBMITTED: 1,
            task.DONE: 1,
            task.EXCEPTION: 1,
            PynisherPlugin.TIMEOUT: 1,
            PynisherPlugin.CPU_TIME_LIMIT_REACHED: 1,
        },
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.STOP: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
            scheduler.FUTURE_EXCEPTION: 1,
        },
    )


def test_cpu_exception_can_be_ignored_by_scheduler(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("cpu_time"):
        pytest.skip("Pynisher does not support cpu time limits on this system")

    task = scheduler.task(
        cpu_time_wasting_function,
        plugins=PynisherPlugin(cputime_limit=1),
    )

    @scheduler.on_start
    def start_task() -> None:
        task.submit(iterations=int(1e16))

    # Should not raise and instead just end and return the exception
    end_status = scheduler.run(
        on_exception={PynisherPlugin.PynisherException: "end"},
    )
    assert end_status.code == ExitState.Code.EXCEPTION
    assert isinstance(end_status.exception, PynisherPlugin.CpuTimeoutException)


def test_memory_limited_task_can_be_ignored_by_scheduler(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("memory"):
        pytest.skip("Pynisher does not support memory limits on this system")

    one_half_gb = int(1e9 * 1.5)
    two_gb = int(1e9) * 2
    task = scheduler.task(
        big_memory_function,
        plugins=PynisherPlugin(memory_limit=one_half_gb),
    )

    @scheduler.on_start
    def start_task() -> None:
        task.submit(mem_in_bytes=two_gb)

    # Should not raise and instead just end and return the exception
    end_status = scheduler.run(
        on_exception={PynisherPlugin.PynisherException: "end"},
    )
    assert end_status.code == ExitState.Code.EXCEPTION
    assert isinstance(end_status.exception, PynisherPlugin.MemoryLimitException)


def test_time_limited_task_can_be_ignored_by_scheduler(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("wall_time"):
        pytest.skip("Pynisher does not support wall time limits on this system")

    task = scheduler.task(
        time_wasting_function,
        plugins=PynisherPlugin(walltime_limit=1),
    )

    @scheduler.on_start
    def start_task() -> None:
        task.submit(duration=3)

    # Should ignore the exception and move on
    end_status = scheduler.run(
        on_exception={PynisherPlugin.PynisherException: "end"},
    )
    assert end_status.code == ExitState.Code.EXCEPTION
    assert isinstance(end_status.exception, PynisherPlugin.WallTimeoutException)


def test_trial_gets_autodetect_memory(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("memory"):
        pytest.skip("Pynisher does not support memory limits on this system")

    one_half_gb = int(1e9 * 1.5)
    two_gb = int(1e9) * 2
    task = scheduler.task(
        trial_with_big_memory,
        plugins=PynisherPlugin(
            memory_limit=one_half_gb,
            disable_trial_handling=False,
        ),
    )
    trial = Trial(name="test_trial", config={})

    @scheduler.on_start
    def start_task() -> None:
        task.submit(trial, mem_in_bytes=two_gb)

    reports: list[Trial.Report] = []

    @task.on_result
    def trial_report(_, report: Trial.Report) -> None:
        reports.append(report)

    status = scheduler.run(on_exception="raise")
    assert status.code == ExitState.Code.EXHAUSTED

    assert task.event_counts == Counter(
        {task.SUBMITTED: 1, task.DONE: 1, task.RESULT: 1},
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.FUTURE_RESULT: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
        },
    )
    assert len(reports) == 1
    assert reports[0].status == Trial.Status.FAIL
    assert isinstance(reports[0].exception, PynisherPlugin.MemoryLimitException)


def test_trial_gets_autodetect_time(scheduler: Scheduler) -> None:
    if not PynisherPlugin.supports("wall_time"):
        pytest.skip("Pynisher does not support wall_time limits on this system")

    task = scheduler.task(
        trial_with_time_wasting,
        plugins=PynisherPlugin(
            walltime_limit=1,
            disable_trial_handling=False,
        ),
    )
    trial = Trial(name="test_trial", config={})

    @scheduler.on_start
    def start_task() -> None:
        task.submit(trial=trial, duration=3)

    reports: list[Trial.Report] = []

    @task.on_result
    def trial_report(_, report: Trial.Report) -> None:
        reports.append(report)

    status = scheduler.run(on_exception="raise")
    assert status.code == ExitState.Code.EXHAUSTED

    assert task.event_counts == Counter(
        {task.SUBMITTED: 1, task.DONE: 1, task.RESULT: 1},
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.FUTURE_RESULT: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
        },
    )
    assert len(reports) == 1
    assert reports[0].status == Trial.Status.FAIL
    assert isinstance(reports[0].exception, PynisherPlugin.WallTimeoutException)
