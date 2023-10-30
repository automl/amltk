from __future__ import annotations

import logging
import warnings
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

# We need these imported to ensure that the threadpoolctl plugin
# actually does something.
import numpy  # noqa: F401
import sklearn  # noqa: F401
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

import threadpoolctl
from amltk.scheduling import Scheduler, SequentialExecutor
from amltk.scheduling.scheduler import ExitState
from amltk.threadpoolctl import ThreadPoolCTLPlugin
from amltk.types import safe_isinstance

logger = logging.getLogger(__name__)


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
    return SequentialExecutor()


@case(tags=["executor"])
def case_loky_executor() -> ProcessPoolExecutor:
    from loky import get_reusable_executor

    return get_reusable_executor(max_workers=2)  # type: ignore


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Scheduler:
    return Scheduler(executor)


def f() -> list[Any]:
    return threadpoolctl.threadpool_info()


def test_empty_kwargs_does_not_change_anything(scheduler: Scheduler) -> None:
    task = scheduler.task(f, plugins=ThreadPoolCTLPlugin())

    retrieved_info = []
    before = threadpoolctl.threadpool_info()

    @scheduler.on_start
    def start_task() -> None:
        task()

    @task.on_result
    def check_threadpool_info(_, inner_info: list) -> None:
        retrieved_info.append(inner_info)

    end_status = scheduler.run()
    assert end_status == ExitState(code=Scheduler.ExitCode.EXHAUSTED)

    inside_info = retrieved_info[0]
    after = threadpoolctl.threadpool_info()

    # NOTE: For whatever reason, Loky seems to pick up on different blas library
    # and use that, different from every other executor
    if not safe_isinstance(scheduler.executor, "_ReusablePoolExecutor"):
        assert before == inside_info

    assert before == after

    assert task.event_counts == Counter(
        {task.SUBMITTED: 1, task.DONE: 1, task.RESULT: 1},
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
            scheduler.FUTURE_RESULT: 1,
        },
    )


def test_limiting_thread_count_limits_only_inside_task(scheduler: Scheduler) -> None:
    task = scheduler.task(f, plugins=ThreadPoolCTLPlugin(max_threads=1))

    retrieved_info = []
    before = threadpoolctl.threadpool_info()

    @scheduler.on_start
    def start_task() -> None:
        task()

    @task.on_result
    def check_threadpool_info(_, inner_info: list) -> None:
        retrieved_info.append(inner_info)

    end_status = scheduler.run()
    assert end_status == ExitState(code=Scheduler.ExitCode.EXHAUSTED)

    inside_info = retrieved_info[0]
    after = threadpoolctl.threadpool_info()

    assert before != inside_info
    assert before == after

    assert task.event_counts == Counter(
        {task.SUBMITTED: 1, task.DONE: 1, task.RESULT: 1},
    )

    assert scheduler.event_counts == Counter(
        {
            scheduler.STARTED: 1,
            scheduler.FINISHING: 1,
            scheduler.FINISHED: 1,
            scheduler.EMPTY: 1,
            scheduler.FUTURE_SUBMITTED: 1,
            scheduler.FUTURE_DONE: 1,
            scheduler.FUTURE_RESULT: 1,
        },
    )
