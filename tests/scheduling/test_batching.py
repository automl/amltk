from __future__ import annotations

import logging
import time
import warnings
from asyncio import Future
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor

import pytest
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.scheduling import CallLimiter, Scheduler, SequentialExecutor, Task

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
    pytest.skip("SequentialExecutor does not support batching.")
    return SequentialExecutor()


@case(tags=["executor"])
def case_loky_executor() -> ProcessPoolExecutor:
    from loky import get_reusable_executor

    return get_reusable_executor(max_workers=2)  # type: ignore


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Scheduler:
    return Scheduler(executor)


def test_batch_all_successful(scheduler: Scheduler) -> None:
    task = Task(sleep_and_return, scheduler)
    args = [(0.1,), (0.1,), (0.1,)]
    batch = task.batch(args)

    results = []

    @scheduler.on_start()
    def on_start() -> None:
        batch.submit()

    @batch.on_batch_returned
    def on_batch_returned(_: Task.Batch, _results: list[float]) -> None:
        results.extend(_results)

    end_status = scheduler.run()

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert results == [0.1, 0.1, 0.1]

    assert task.event_counts == {}
    assert batch.event_counts == {
        batch.BATCH_RETURNED: 1,
        batch.BATCH_SUBMITTED: 1,
        batch.BATCH_DONE: 1,
        batch.ANY_RETURNED: len(args),
        batch.ANY_SUBMITTED: len(args),
        batch.ANY_DONE: len(args),
    }
    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.EMPTY: 1,
        scheduler.FINISHED: 1,
        scheduler.FUTURE_SUBMITTED: len(args),
        scheduler.FUTURE_DONE: len(args),
    }


def test_batch_any_returned(scheduler: Scheduler) -> None:
    task = Task(sleep_and_return, scheduler)
    args = [(0.1,), (0.1,), (0.1,)]
    batch = task.batch(args)

    results = []

    @scheduler.on_start()
    def on_start() -> None:
        batch.submit()

    @batch.on_any_returned
    def on_batch_returned(_: Task.Batch, _results: float) -> None:
        results.append(_results)

    end_status = scheduler.run()

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert results == [0.1, 0.1, 0.1]

    assert task.event_counts == {}
    assert batch.event_counts == {
        batch.BATCH_RETURNED: 1,
        batch.BATCH_SUBMITTED: 1,
        batch.BATCH_DONE: 1,
        batch.ANY_RETURNED: len(args),
        batch.ANY_SUBMITTED: len(args),
        batch.ANY_DONE: len(args),
    }
    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: len(args),
        scheduler.FUTURE_DONE: len(args),
    }


def test_batch_any_exception(scheduler: Scheduler) -> None:
    task = Task(raise_exception, scheduler)
    args = [(), (), ()]
    batch = task.batch(args)

    results = []

    @scheduler.on_start()
    def on_start() -> None:
        batch.submit()

    @batch.on_any_exception
    def on_any_exception(_: Task.Batch, exception: BaseException) -> None:
        results.append(exception)

    end_status = scheduler.run(end_on_exception=False)

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert len(results) == len(args)

    assert task.event_counts == {}
    assert batch.event_counts == {
        batch.BATCH_SUBMITTED: 1,
        batch.BATCH_DONE: 1,
        batch.ANY_SUBMITTED: len(args),
        batch.ANY_EXCEPTION: len(args),
        batch.ANY_DONE: len(args),
    }
    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: len(args),
        scheduler.FUTURE_DONE: len(args),
    }


def test_batch_any_cancelled(scheduler: Scheduler) -> None:
    task = Task(sleep_and_return, scheduler)
    args = [(0.5,), (0.5,), (0.5,)]

    batch = task.batch(args)

    results = []

    @scheduler.on_start()
    def on_start() -> None:
        batch.submit()

    # Make sure to cancel the batch when the first task is submitted
    @batch.on_any_submitted
    def on_any_submitted(_: Task.Batch, fut: Future[float]) -> None:  # noqa: ARG001
        batch.cancel()

    # None of the tasks should be submitted
    @batch.on_any_returned
    def on_any_returned(_: Task.Batch, _results: float) -> None:
        results.append(_results)

    end_status = scheduler.run()

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert results == []

    assert task.event_counts == {}
    assert batch.event_counts == {
        batch.BATCH_CANCELLED: 1,
        batch.ANY_SUBMITTED: 1,
    }

    expected_scheduler_event_counts = {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_CANCELLED: 1,
    }

    assert scheduler.event_counts == expected_scheduler_event_counts


def test_batch_task_does_not_share_events_with_single_task(
    scheduler: Scheduler,
) -> None:
    task = Task(sleep_and_return, scheduler)
    args = [(0.1,)]

    batch = task.batch(args)

    batch_results = []
    task_results = []

    @scheduler.on_start()
    def on_start() -> None:
        batch.submit()

        # Here we make sure to use the task to submit
        task.submit(0.2)

    @batch.on_any_returned
    def on_any_returned(_: Task.Batch, x: float) -> None:
        batch_results.append(x)

    @task.on_returned
    def on_returned(_, x: float) -> None:
        task_results.append(x)

    end_status = scheduler.run()

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert batch_results == [0.1]
    assert task_results == [0.2]

    assert task.event_counts == {task.SUBMITTED: 1, task.DONE: 1, task.RETURNED: 1}
    assert batch.event_counts == {
        batch.BATCH_SUBMITTED: 1,
        batch.BATCH_DONE: 1,
        batch.BATCH_RETURNED: 1,
        batch.ANY_SUBMITTED: 1,
        batch.ANY_RETURNED: 1,
        batch.ANY_DONE: 1,
    }
    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 2,
        scheduler.FUTURE_DONE: 2,
    }


def test_batch_stop_on_failed_submission(scheduler: Scheduler) -> None:
    # The scheduler has 2 workers so we expect the first to go through,
    # the second to not be not submitted due to the plugin, and hence the remaininig
    # tasks to not be submitted.
    task = Task[[float], float](
        sleep_and_return,
        scheduler,
        plugins=[CallLimiter(max_calls=1)],
    )

    args = [(1.0,), (1.0,), (1.0,), (1.0,)]
    batch = task.batch(args)

    submitted_futures = []

    @scheduler.on_start
    def on_start() -> None:
        submissions = batch.submit()
        submitted_futures.extend(submissions)

    end_status = scheduler.run()

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert len(submitted_futures) == 1

    assert task.event_counts == {}
    assert batch.event_counts == {
        batch.BATCH_FAILED: 1,
        batch.ANY_SUBMITTED: 1,
        batch.BATCH_CANCELLED: 1,
    }

    expected_scheduler_event_counts = {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_CANCELLED: 1,
    }

    assert scheduler.event_counts == expected_scheduler_event_counts
