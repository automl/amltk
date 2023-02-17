"""TODO: Rework these tests once scheduler complete."""
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time

from dask.distributed import Client, LocalCluster
from distributed.cfexecutor import ClientExecutor
import pytest
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Scheduler, SchedulerEvent, TaskEvent

logger = logging.getLogger(__name__)


def sleep_and_return(sleep_time: float) -> float:
    time.sleep(sleep_time)
    logger.debug(f"Done sleep for {sleep_time}!")
    return sleep_time


@case(tags=["executor"])
def case_thread_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_process_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_dask_executor() -> ClientExecutor:
    # NOTE: There is logging errors for the workers shutting down
    # but we can safely ignore it. One limitation might be that
    # other dask errors get hidden by this.
    cluster = LocalCluster(n_workers=2, silence_logs=logging.ERROR)
    client = Client(cluster)
    return client.get_executor()


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Scheduler:
    # HACK: This is a hack, so we can yield and clean up
    return Scheduler(executor)


def test_scheduler_with_timeout_and_wait_for_tasks(scheduler: Scheduler) -> None:
    results: list[float] = []
    sleep_time = 0.5

    task = scheduler.task("sleep", sleep_and_return, sleep_time=sleep_time)
    task.on_success(results.append)
    end_status = scheduler.run(task, timeout=0.1, wait=True)

    assert results == [sleep_time]

    task_expected_counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.FINISHED: 1,
        TaskEvent.SUCCESS: 1,
        TaskEvent.CANCELLED: 0,
        TaskEvent.ERROR: 0,
        TaskEvent.UPDATE: 0,
        TaskEvent.WAITING: 0,
    }

    assert task.event_counts == task_expected_counts

    assert scheduler.event_counts == task_expected_counts

    expected_status_counts = {
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.STOPPING: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.TIMEOUT: 1,
        SchedulerEvent.STOPPED: 0,
        SchedulerEvent.EMPTY: 0,
    }

    assert scheduler.status_counts == expected_status_counts

    assert end_status == scheduler.exitcode.TIMEOUT
    assert scheduler.empty()
    assert not scheduler.running()


def test_scheduler_with_timeout_and_not_wait_for_tasks(scheduler: Scheduler) -> None:
    if isinstance(scheduler.executor, ThreadPoolExecutor):
        pytest.skip(
            "There is no forcibul way to kill a thread, so this test will hang"
            " for the full `sleep_time`. While technically this will pass the"
            " test, we do not want this hanging behaviour. This should be"
            " explicitly documented when using ThreadPoolExecutor."
        )

    results: list[float] = []
    sleep_time = 10
    task = scheduler.task("sleep", sleep_and_return, sleep_time=sleep_time).on_success(
        results.append
    )
    end_status = scheduler.run(task, timeout=0.1, wait=False)

    # No results collect as task cancelled
    assert results == []

    task_expected_counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.CANCELLED: 1,
        TaskEvent.FINISHED: 0,
        TaskEvent.SUCCESS: 0,
        TaskEvent.ERROR: 0,
        TaskEvent.UPDATE: 0,
        TaskEvent.WAITING: 0,
    }
    assert task.event_counts == task_expected_counts
    assert scheduler.event_counts == task_expected_counts

    expected_status_counts = {
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.STOPPING: 1,
        SchedulerEvent.TIMEOUT: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.STOPPED: 0,
        SchedulerEvent.EMPTY: 0,
    }
    assert scheduler.status_counts == expected_status_counts

    assert end_status == scheduler.exitcode.TIMEOUT
    assert scheduler.empty()
    assert not scheduler.running()


def test_chained_tasks(scheduler: Scheduler) -> None:
    results: list[float] = []
    second_task = scheduler.task("second", sleep_and_return, 0.1)
    second_task.on_success(results.append)

    first_task = (
        scheduler.task("first", sleep_and_return, 0.1)
        .on_success(results.append)
        .on_success(second_task)
    )
    end_status = scheduler.run(first_task, end_on_empty=True)

    assert end_status == scheduler.exitcode.EMPTY
    assert results == [0.1, 0.1]

    expected_task_event_counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.SUCCESS: 1,
        TaskEvent.FINISHED: 1,
        TaskEvent.CANCELLED: 0,
        TaskEvent.ERROR: 0,
        TaskEvent.UPDATE: 0,
        TaskEvent.WAITING: 0,
    }
    assert first_task.event_counts == expected_task_event_counts
    assert second_task.event_counts == expected_task_event_counts

    expected_global_task_event_counts = {
        TaskEvent.SUBMITTED: 2,
        TaskEvent.SUCCESS: 2,
        TaskEvent.FINISHED: 2,
        TaskEvent.CANCELLED: 0,
        TaskEvent.ERROR: 0,
        TaskEvent.UPDATE: 0,
        TaskEvent.WAITING: 0,
    }
    assert scheduler.event_counts == expected_global_task_event_counts

    expected_status_counts = {
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.STOPPING: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.TIMEOUT: 0,
        SchedulerEvent.STOPPED: 0,
        SchedulerEvent.EMPTY: 0,
    }
    assert scheduler.status_counts == expected_status_counts
    assert scheduler.empty()
    assert not scheduler.running()


def test_queue_empty_status(scheduler: Scheduler) -> None:
    task = scheduler.task("sleep", sleep_and_return, sleep_time=0.1)

    # Reload on the first empty
    scheduler.on_empty(
        task, when=lambda _: scheduler.status_counts[SchedulerEvent.EMPTY] == 1
    )

    # Stop on the second empty
    scheduler.on_empty(
        scheduler.stop,
        when=lambda _: scheduler.status_counts[SchedulerEvent.EMPTY] == 2,
        name="stop",
    )

    end_status = scheduler.run(task, timeout=3, end_on_empty=False)

    assert scheduler.status_counts[SchedulerEvent.EMPTY] == 2
    assert end_status == scheduler.exitcode.STOPPED
    assert scheduler.empty()
