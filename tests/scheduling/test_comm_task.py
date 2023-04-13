from __future__ import annotations

import logging
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Hashable, Iterator

from dask.distributed import Client, LocalCluster, Worker
from distributed.cfexecutor import ClientExecutor
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Comm, CommTask, Scheduler

logger = logging.getLogger(__name__)


def sending_worker(comm: Comm, replies: list[Any]) -> None:
    """A worker that responds to messages.

    Args:
        comm: The communication channel to use.
        replies: A list of replies to send to the client.
    """
    with comm:
        for reply in replies:
            comm.send(reply)


def requesting_worker(comm: Comm, requests: list[Any]) -> None:
    """A worker that waits for messages.

    This will send a request, waiting for a response, finally
    sending a msg of the response recieved.
    sending

    Args:
        comm: The communication channel to use.
        requests: A list of requests to receive from the client.
    """
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

    task = CommTask(sending_worker, scheduler=scheduler)

    @task.on_message
    def handle_msg(msg: Any) -> None:
        results.append(msg.data)

    @scheduler.on_start
    def start() -> None:
        task(replies)

    end_status = scheduler.run()

    task_counts: dict[Hashable, int] = {
        CommTask.SUBMITTED: 1,
        CommTask.DONE: 1,
        CommTask.RETURNED: 1,
        CommTask.MESSAGE: len(replies),
        CommTask.F_RETURNED: 1,
        CommTask.CLOSE: 1,
    }
    assert task.counts == task_counts

    assert end_status == Scheduler.ExitCode.EXHAUSTED
    scheduler_counts = {
        (CommTask.SUBMITTED, "sending_worker"): 1,
        (CommTask.DONE, "sending_worker"): 1,
        (CommTask.RETURNED, "sending_worker"): 1,
        (CommTask.MESSAGE, "sending_worker"): len(replies),
        (CommTask.CLOSE, "sending_worker"): 1,
        (CommTask.F_RETURNED, "sending_worker"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
    assert scheduler.counts == scheduler_counts
    assert results == [1, 2, 3]


def test_waiting_worker(scheduler: Scheduler) -> None:
    """Test that the scheduler can receive replies."""
    requests = [1, 2, 3]
    results: list[int] = []

    task = CommTask(requesting_worker, scheduler=scheduler)

    @task.on_request
    def handle_waiting(msg: CommTask.Msg) -> None:
        msg.respond(msg.data * 2)

    @task.on_message
    def handle_msg(msg: CommTask.Msg) -> None:
        results.append(msg.data)

    @scheduler.on_start
    def start() -> None:
        task(requests)

    end_status = scheduler.run()
    assert end_status == Scheduler.ExitCode.EXHAUSTED

    assert results == [2, 4, 6]

    assert task.counts == {
        CommTask.SUBMITTED: 1,
        CommTask.DONE: 1,
        CommTask.RETURNED: 1,
        CommTask.F_RETURNED: 1,
        CommTask.MESSAGE: len(results),
        CommTask.REQUEST: len(requests),
        CommTask.CLOSE: 1,
    }

    assert scheduler.counts == {
        (CommTask.SUBMITTED, "requesting_worker"): 1,
        (CommTask.DONE, "requesting_worker"): 1,
        (CommTask.RETURNED, "requesting_worker"): 1,
        (CommTask.F_RETURNED, "requesting_worker"): 1,
        (CommTask.MESSAGE, "requesting_worker"): len(results),
        (CommTask.REQUEST, "requesting_worker"): len(requests),
        (CommTask.CLOSE, "requesting_worker"): 1,
        Scheduler.STARTED: 1,
        Scheduler.FINISHING: 1,
        Scheduler.FINISHED: 1,
    }
