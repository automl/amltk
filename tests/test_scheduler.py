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

    task = scheduler.task(sleep_and_return, name="sleep")

    task.on_return(results.append)

    def begin() -> None:
        task(sleep_time=sleep_time)

    scheduler.on_start(begin)
    end_status = scheduler.run(timeout=0.1, wait=True)

    assert results == [sleep_time]

    counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.DONE: 1,
        TaskEvent.RETURNED: 1,
        ("sleep", TaskEvent.SUBMITTED): 1,
        ("sleep", TaskEvent.DONE): 1,
        ("sleep", TaskEvent.RETURNED): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.TIMEOUT: 1,
    }
    assert dict(scheduler.counts) == counts
    assert end_status == scheduler.exitcodes.TIMEOUT
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
    task = scheduler.task(sleep_and_return, name="sleep")
    scheduler.on_start(lambda: task(sleep_time=10))

    end_status = scheduler.run(timeout=0.1, wait=False)

    # No results collect as task cancelled
    assert results == []

    expected_counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.CANCELLED: 1,
        ("sleep", TaskEvent.SUBMITTED): 1,
        ("sleep", TaskEvent.CANCELLED): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.TIMEOUT: 1,
    }

    assert scheduler.counts == expected_counts
    assert end_status == scheduler.exitcodes.TIMEOUT
    assert scheduler.empty()
    assert not scheduler.running()


def test_chained_tasks(scheduler: Scheduler) -> None:
    results: list[float] = []
    task_1 = scheduler.task(sleep_and_return, name="first")
    task_2 = scheduler.task(sleep_and_return, name="second")
    task_1.on_return(results.append)
    task_2.on_return(results.append)

    # Feed the output of task_1 into task_2
    task_1.on_return(task_2)
    scheduler.on_start(lambda: task_1(sleep_time=0.1))

    end_status = scheduler.run(wait=True)

    expected_counts = {
        TaskEvent.SUBMITTED: 2,
        TaskEvent.DONE: 2,
        TaskEvent.RETURNED: 2,
        ("first", TaskEvent.SUBMITTED): 1,
        ("first", TaskEvent.DONE): 1,
        ("first", TaskEvent.RETURNED): 1,
        ("second", TaskEvent.SUBMITTED): 1,
        ("second", TaskEvent.DONE): 1,
        ("second", TaskEvent.RETURNED): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
    }
    assert scheduler.counts == expected_counts
    assert results == [0.1, 0.1]
    assert end_status == scheduler.exitcodes.EXHAUSTED
    assert scheduler.empty()
    assert not scheduler.running()


def test_queue_empty_status(scheduler: Scheduler) -> None:
    task = scheduler.task(sleep_and_return, name="sleep")

    # Reload on the first empty
    scheduler.on_empty(
        lambda: task(sleep_time=0.1),
        when=lambda counts: counts[SchedulerEvent.EMPTY] == 1,
    )

    # Stop on the second empty
    scheduler.on_empty(
        scheduler.stop,
        when=lambda counts: counts[SchedulerEvent.EMPTY] == 2,
    )

    end_status = scheduler.run(timeout=3, end_on_empty=False)

    expected_counts = {
        TaskEvent.SUBMITTED: 1,
        TaskEvent.DONE: 1,
        TaskEvent.RETURNED: 1,
        ("sleep", TaskEvent.SUBMITTED): 1,
        ("sleep", TaskEvent.DONE): 1,
        ("sleep", TaskEvent.RETURNED): 1,
        SchedulerEvent.STARTED: 1,
        SchedulerEvent.FINISHING: 1,
        SchedulerEvent.FINISHED: 1,
        SchedulerEvent.EMPTY: 2,
        SchedulerEvent.STOP: 1,
    }

    assert scheduler.counts == expected_counts
    assert end_status == scheduler.exitcodes.STOPPED
    assert scheduler.empty()
