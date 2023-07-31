from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Iterator

from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.scheduling import Scheduler, Task
from amltk.scheduling.task_plugin import CallLimiter


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


def time_wasting_function(duration: int) -> int:
    time.sleep(duration)
    return duration


def test_concurrency_limit_of_tasks(scheduler: Scheduler) -> None:
    limiter = CallLimiter(max_concurrent=2)
    task = Task(  # type: ignore
        time_wasting_function,
        scheduler=scheduler,
        plugins=[limiter],
    )

    @scheduler.on_start(repeat=10)
    def launch_many() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.event_counts == {
        task.SUBMITTED: 2,
        task.F_SUBMITTED: 2,
        task.DONE: 2,
        task.RETURNED: 2,
        task.F_RETURNED: 2,
    }
    assert limiter.event_counts == {limiter.CONCURRENT_LIMIT_REACHED: 8}

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 2,
        scheduler.FUTURE_DONE: 2,
    }


def test_call_limit_of_tasks(scheduler: Scheduler) -> None:
    limiter = CallLimiter(max_calls=2)
    task = Task(  # type: ignore
        time_wasting_function,
        scheduler,
        plugins=[limiter],
    )

    @scheduler.on_start(repeat=10)
    def launch() -> None:
        task(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task.event_counts == {
        task.SUBMITTED: 2,
        task.F_SUBMITTED: 2,
        task.DONE: 2,
        task.RETURNED: 2,
        task.F_RETURNED: 2,
    }
    assert limiter.event_counts == {limiter.CALL_LIMIT_REACHED: 8}

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 2,
        scheduler.FUTURE_DONE: 2,
    }


def test_call_limit_with_not_while_running(scheduler: Scheduler) -> None:
    task1 = Task(  # type: ignore
        time_wasting_function,
        scheduler,
    )

    limiter = CallLimiter(not_while_running=task1)
    task2 = Task(  # type: ignore
        time_wasting_function,
        scheduler,
        plugins=[limiter],
    )

    @scheduler.on_start()
    def launch() -> None:
        task1(duration=2)

    @task1.on_submitted
    def launch2(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        task2(duration=2)

    end_status = scheduler.run(end_on_empty=True)
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert task1.event_counts == {
        task1.SUBMITTED: 1,
        task1.F_SUBMITTED: 1,
        task1.DONE: 1,
        task1.RETURNED: 1,
        task1.F_RETURNED: 1,
    }

    assert limiter.event_counts == {limiter.DISABLED_DUE_TO_RUNNING_TASK: 1}
    assert task2.event_counts == {}

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.EMPTY: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }
