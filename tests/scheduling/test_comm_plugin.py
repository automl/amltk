from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Hashable, Iterator

import pytest
from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from amltk.scheduling import Scheduler, Task
from amltk.scheduling.comms import Comm

logger = logging.getLogger(__name__)


def sending_worker(replies: list[Any], comm: Comm | None = None) -> None:
    """A worker that responds to messages.

    Args:
        comm: The communication channel to use.
        replies: A list of replies to send to the client.
    """
    assert comm is not None

    with comm:
        for reply in replies:
            comm.send(reply)


def requesting_worker(requests: list[Any], comm: Comm | None = None) -> None:
    """A worker that waits for messages.

    This will send a request, waiting for a response, finally
    sending a msg of the response recieved.
    sending

    Args:
        comm: The communication channel to use.
        requests: A list of requests to receive from the client.
    """
    assert comm is not None

    with comm:
        for request in requests:
            response = comm.request(request)
            comm.send(response)


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
    pytest.skip(
        "Dask executor stopped support for passing Connection" " objects in 2023.4",
    )
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


def test_sending_worker(scheduler: Scheduler) -> None:
    """Test that the scheduler can receive replies."""
    replies = [1, 2, 3]
    results: list[int] = []

    comm_plugin = Comm.Plugin()
    task = Task(sending_worker, scheduler=scheduler, plugins=[comm_plugin])

    @task.on(Comm.MESSAGE)
    def handle_msg(msg: Any) -> None:
        results.append(msg.data)

    @scheduler.on_start
    def start() -> None:
        task(replies)

    end_status = scheduler.run()

    task_counts: dict[Hashable, int] = {
        task.SUBMITTED: 1,
        task.F_SUBMITTED: 1,
        task.DONE: 1,
        task.RETURNED: 1,
        Comm.MESSAGE: len(replies),
        task.F_RETURNED: 1,
        Comm.CLOSE: 1,
    }
    assert task.event_counts == task_counts

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    scheduler_counts = {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }
    assert scheduler.event_counts == scheduler_counts
    assert results == [1, 2, 3]


def test_waiting_worker(scheduler: Scheduler) -> None:
    """Test that the scheduler can receive replies."""
    requests = [1, 2, 3]
    results: list[int] = []

    comm_plugin = Comm.Plugin()
    task = Task(requesting_worker, scheduler=scheduler, plugins=[comm_plugin])

    @task.on(Comm.REQUEST)
    def handle_waiting(msg: Comm.Msg) -> None:
        msg.respond(msg.data * 2)

    @task.on(Comm.MESSAGE)
    def handle_msg(msg: Comm.Msg) -> None:
        results.append(msg.data)

    @scheduler.on_start
    def start() -> None:
        task(requests)

    end_status = scheduler.run()
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert results == [2, 4, 6]

    assert task.event_counts == {
        task.SUBMITTED: 1,
        task.F_SUBMITTED: 1,
        task.DONE: 1,
        task.RETURNED: 1,
        task.F_RETURNED: 1,
        Comm.MESSAGE: len(results),
        Comm.REQUEST: len(requests),
        Comm.CLOSE: 1,
    }

    assert scheduler.event_counts == {
        scheduler.STARTED: 1,
        scheduler.FINISHING: 1,
        scheduler.FINISHED: 1,
        scheduler.FUTURE_SUBMITTED: 1,
        scheduler.FUTURE_DONE: 1,
    }
