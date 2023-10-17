from __future__ import annotations

import logging
import time
import warnings
from asyncio import Future
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Hashable

import pytest
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.scheduling import ExitState, Scheduler, SequentialExecutor, Task
from amltk.types import safe_isinstance

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CustomError(Exception):
    """A custom error for testing."""


def sleep_and_return(sleep_time: float) -> float:
    time.sleep(sleep_time)
    logger.debug(f"Done sleep for {sleep_time}!")
    return sleep_time


def raise_exception() -> None:
    raise CustomError("This is a custom error.")


@case(tags=["executor"])
def case_thread_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_process_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=2)


@case(tags=["executor"])
def case_loky() -> ProcessPoolExecutor:
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
    return client.get_executor()


@case(tags=["executor"])
def case_sequential_executor() -> SequentialExecutor:
    return SequentialExecutor()


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Scheduler:
    return Scheduler(executor)


def test_scheduler_with_timeout_and_wait_for_tasks(scheduler: Scheduler) -> None:
    if isinstance(scheduler.executor, SequentialExecutor):
        pytest.skip(
            "SequentialExecutor will complete the task but the scheduler will"
            " not realise it's a timeout.",
        )

    results: list[float] = []
    sleep_time = 0.5

    task = Task(sleep_and_return, scheduler, name="sleep")

    @task.on_returned
    def append_result(_: Future, res: float) -> None:
        results.append(res)

    @scheduler.on_start
    def launch_task() -> None:
        task(sleep_time=sleep_time)

    end_status = scheduler.run(timeout=0.1, wait=True)
    assert results == [sleep_time]

    assert task.event_counts == {task.SUBMITTED: 1, task.DONE: 1, task.RETURNED: 1}

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.TIMEOUT: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }
    assert end_status == ExitState(code=Scheduler.ExitCode.TIMEOUT)
    assert scheduler.empty()
    assert not scheduler.running()


def test_scheduler_with_timeout_and_not_wait_for_tasks(scheduler: Scheduler) -> None:
    if isinstance(scheduler.executor, ThreadPoolExecutor):
        pytest.skip(
            "There is no forcibul way to kill a thread, so this test will hang"
            " for the full `sleep_time`. While technically this will pass the"
            " test, we do not want this hanging behaviour. This should be"
            " explicitly documented when using ThreadPoolExecutor.",
        )

    if isinstance(scheduler.executor, SequentialExecutor):
        pytest.skip("SequentialExecutor can not be interupted while a task is running.")

    results: list[float] = []
    task = Task(sleep_and_return, scheduler, name="sleep")
    scheduler.on_start(lambda: task(sleep_time=10))

    end_status = scheduler.run(timeout=0.1, wait=False)

    # No results collect as task cancelled
    assert results == []

    # We have a termination strategy for ProcessPoolExecutor and we know it
    expected_task_counts = {task.SUBMITTED: 1, task.CANCELLED: 1}

    assert task.event_counts == expected_task_counts

    expected_scheduler_counts: dict[Hashable, int] = {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.TIMEOUT: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }

    # NOTE: This is because Dask can cancel currently running tasks which is not
    # something that can be done with Python's default executors.
    if isinstance(
        scheduler.executor,
        (ClientExecutor, ProcessPoolExecutor),
    ) or safe_isinstance(scheduler.executor, "_ReusablePoolExecutor"):
        expected_scheduler_counts[scheduler.FUTURE_CANCELLED] = 1
        del expected_scheduler_counts[scheduler.FUTURE_DONE]

    assert scheduler.event_counts == expected_scheduler_counts
    assert end_status == ExitState(code=Scheduler.ExitCode.TIMEOUT)
    assert scheduler.empty()
    assert not scheduler.running()


def test_chained_tasks(scheduler: Scheduler) -> None:
    results: list[float] = []
    task_1 = Task(sleep_and_return, scheduler, name="first")
    task_2 = Task(sleep_and_return, scheduler, name="second")

    # Feed the output of task_1 into task_2
    task_1.on_returned(lambda _, res: task_2(sleep_time=res))
    task_1.on_returned(lambda _, res: results.append(res))
    task_2.on_returned(lambda _, res: results.append(res))

    scheduler.on_start(lambda: task_1(sleep_time=0.1))

    end_status = scheduler.run(wait=True)

    expected_counts = {task_1.SUBMITTED: 1, task_1.DONE: 1, task_1.RETURNED: 1}
    assert task_1.event_counts == expected_counts
    assert task_2.event_counts == expected_counts

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 2,
        scheduler.FUTURE_DONE: 2,
    }
    assert results == [0.1, 0.1]
    assert end_status == ExitState(code=Scheduler.ExitCode.EXHAUSTED)
    assert scheduler.empty()
    assert not scheduler.running()


def test_queue_empty_status(scheduler: Scheduler) -> None:
    task = Task(sleep_and_return, scheduler, name="sleep")

    # Reload on the first empty
    @scheduler.on_empty(when=lambda: scheduler.event_counts[scheduler.EMPTY] == 1)
    def launch_first() -> None:
        task(sleep_time=0.1)

    # Stop on the second empty
    @scheduler.on_empty(when=lambda: scheduler.event_counts[scheduler.EMPTY] == 2)
    def stop_scheduler() -> None:
        scheduler.stop()

    end_status = scheduler.run(timeout=3, end_on_empty=False)

    assert task.event_counts == {task.SUBMITTED: 1, task.DONE: 1, task.RETURNED: 1}

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 2,
        scheduler.STOP: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }
    assert end_status == ExitState(code=Scheduler.ExitCode.STOPPED)
    assert scheduler.empty()


def test_repeat_on_start(scheduler: Scheduler) -> None:
    results: list[float] = []

    @scheduler.on_start(repeat=10)
    def append_1() -> None:
        results.append(1)

    end_status = scheduler.run()

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
    }
    assert end_status == ExitState(code=Scheduler.ExitCode.EXHAUSTED)
    assert scheduler.empty()
    assert results == [1] * 10


def test_raise_on_exception_in_task(scheduler: Scheduler) -> None:
    task = Task(raise_exception, scheduler, name="raise_error")

    @scheduler.on_start
    def run_task() -> None:
        task()

    with pytest.raises(CustomError):
        scheduler.run(on_exception="raise")


def test_end_on_exception_in_task(scheduler: Scheduler) -> None:
    task = Task(raise_exception, scheduler, name="raise_error")

    @scheduler.on_start
    def run_task() -> None:
        task()

    end_status = scheduler.run(on_exception="end")
    assert end_status.code == Scheduler.ExitCode.EXCEPTION
    assert isinstance(end_status.exception, CustomError)


def test_dont_end_on_exception_in_task(scheduler: Scheduler) -> None:
    task = Task(raise_exception, scheduler, name="raise_error")

    @scheduler.on_start
    def run_task() -> None:
        task()

    end_status = scheduler.run(on_exception="ignore")
    assert end_status.code == Scheduler.ExitCode.EXHAUSTED
