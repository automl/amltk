from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
import logging
import time
import warnings

from dask.distributed import Client, LocalCluster
from distributed.cfexecutor import ClientExecutor
import pytest
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Scheduler, Task

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
    # Dask will raise a warning when re-using the ports, hence
    # we silence the warnings here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster = LocalCluster(n_workers=2, silence_logs=logging.ERROR, processes=False)

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

    task = Task(sleep_and_return, scheduler, name="sleep")

    @task.on_returned
    def append_result(res: float) -> None:
        results.append(res)

    @scheduler.on_start
    def launch_task() -> None:
        task(sleep_time=sleep_time)

    end_status = scheduler.run(timeout=0.1, wait=True)
    assert results == [sleep_time]

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.RETURNED: 1,
        Task.F_RETURNED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "sleep"): 1,
        (Task.DONE, "sleep"): 1,
        (Task.RETURNED, "sleep"): 1,
        (Task.F_RETURNED, "sleep"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
        Scheduler.TIMEOUT: 1,
    }
    assert end_status == Scheduler.ExitCode.TIMEOUT
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
    task = Task(sleep_and_return, scheduler, name="sleep")
    scheduler.on_start(lambda: task(sleep_time=10))

    end_status = scheduler.run(timeout=0.1, wait=False)

    # No results collect as task cancelled
    assert results == []

    assert task.counts == {Task.SUBMITTED: 1, Task.CANCELLED: 1}

    assert scheduler.counts == {
        (Task.SUBMITTED, "sleep"): 1,
        (Task.CANCELLED, "sleep"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
        Scheduler.TIMEOUT: 1,
    }
    assert end_status == Scheduler.ExitCode.TIMEOUT
    assert scheduler.empty()
    assert not scheduler.running()


def test_chained_tasks(scheduler: Scheduler) -> None:
    results: list[float] = []
    task_1 = Task(sleep_and_return, scheduler, name="first")
    task_2 = Task(sleep_and_return, scheduler, name="second")

    # Feed the output of task_1 into task_2
    task_1.on_returned(lambda res: task_2(sleep_time=res))
    task_1.on_returned(lambda res: results.append(res))
    task_2.on_returned(lambda res: results.append(res))

    scheduler.on_start(lambda: task_1(sleep_time=0.1))

    end_status = scheduler.run(wait=True)

    expected_counts = {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.RETURNED: 1,
        Task.F_RETURNED: 1,
    }
    assert task_1.counts == expected_counts
    assert task_2.counts == expected_counts

    assert scheduler.counts == {
        (Task.SUBMITTED, "first"): 1,
        (Task.DONE, "first"): 1,
        (Task.RETURNED, "first"): 1,
        (Task.F_RETURNED, "first"): 1,
        (Task.SUBMITTED, "second"): 1,
        (Task.DONE, "second"): 1,
        (Task.RETURNED, "second"): 1,
        (Task.F_RETURNED, "second"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
    assert results == [0.1, 0.1]
    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert not scheduler.running()


def test_queue_empty_status(scheduler: Scheduler) -> None:
    task = Task(sleep_and_return, scheduler, name="sleep")

    # Reload on the first empty
    @scheduler.on_empty(when=lambda: scheduler.counts[Scheduler.EMPTY] == 1)
    def launch_first() -> None:
        task(sleep_time=0.1)

    # Stop on the second empty
    @scheduler.on_empty(when=lambda: scheduler.counts[Scheduler.EMPTY] == 2)
    def stop_scheduler() -> None:
        scheduler.stop()

    end_status = scheduler.run(timeout=3, end_on_empty=False)

    assert task.counts == {
        Task.SUBMITTED: 1,
        Task.DONE: 1,
        Task.RETURNED: 1,
        Task.F_RETURNED: 1,
    }

    assert scheduler.counts == {
        (Task.SUBMITTED, "sleep"): 1,
        (Task.DONE, "sleep"): 1,
        (Task.RETURNED, "sleep"): 1,
        (Task.F_RETURNED, "sleep"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
        Scheduler.EMPTY: 2,
        Scheduler.STOP: 1,
    }
    assert end_status == Scheduler.ExitCode.STOPPED
    assert scheduler.empty()


def test_repeat_on_start(scheduler: Scheduler) -> None:
    results: list[float] = []

    @scheduler.on_start(repeat=10)
    def append_1() -> None:
        results.append(1)

    end_status = scheduler.run()

    assert scheduler.counts == {
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
    assert end_status == Scheduler.ExitCode.EXHAUSTED
    assert scheduler.empty()
    assert results == [1] * 10
