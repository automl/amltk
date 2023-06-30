from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

import numpy  # noqa: F401
import sklearn  # noqa: F401
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

# We need these imported to ensure that the threadpoolctl plugin
# actually does something.
import threadpoolctl
from amltk.scheduling import Scheduler, SequentialExecutor, Task
from amltk.threadpoolctl import ThreadPoolCTLPlugin

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


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor) -> Scheduler:
    return Scheduler(executor)


def f() -> list[Any]:
    return threadpoolctl.threadpool_info()


def test_empty_kwargs_does_not_change_anything(scheduler: Scheduler) -> None:
    task: Task[[], list[Any]] = Task(
        f,
        scheduler,
        plugins=[ThreadPoolCTLPlugin()],
    )

    retrieved_info = []
    before = threadpoolctl.threadpool_info()

    @scheduler.on_start
    def start_task() -> None:
        task()

    @task.on_returned
    def check_threadpool_info(inner_info: list) -> None:
        retrieved_info.append(inner_info)

    end_status = scheduler.run()
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    inside_info = retrieved_info[0]
    after = threadpoolctl.threadpool_info()

    assert before == inside_info
    assert before == after

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.F_SUBMITTED: 1,
        Task.DONE: 1,
        Task.RETURNED: 1,
        Task.F_RETURNED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "f"): 1,
        (Task.F_SUBMITTED, "f"): 1,
        (Task.DONE, "f"): 1,
        (Task.RETURNED, "f"): 1,
        (Task.F_RETURNED, "f"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }


def test_limiting_thread_count_limits_only_inside_task(scheduler: Scheduler) -> None:
    task: Task[[], list[Any]] = Task(
        f,
        scheduler,
        plugins=[ThreadPoolCTLPlugin(max_threads=1)],
    )

    retrieved_info = []
    before = threadpoolctl.threadpool_info()

    @scheduler.on_start
    def start_task() -> None:
        task()

    @task.on_returned
    def check_threadpool_info(inner_info: list) -> None:
        retrieved_info.append(inner_info)

    end_status = scheduler.run()
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    inside_info = retrieved_info[0]
    after = threadpoolctl.threadpool_info()

    assert before != inside_info
    assert before == after

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.F_SUBMITTED: 1,
        Task.DONE: 1,
        Task.RETURNED: 1,
        Task.F_RETURNED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "f"): 1,
        (Task.F_SUBMITTED, "f"): 1,
        (Task.DONE, "f"): 1,
        (Task.RETURNED, "f"): 1,
        (Task.F_RETURNED, "f"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
