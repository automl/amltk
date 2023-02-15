"""TODO: Rework these tests once scheduler complete."""
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import time
from typing import Callable, Iterator

from dask.distributed import Client, LocalCluster
from pytest_cases import case, fixture, parametrize_with_cases

from byop.scheduling import Scheduler, SchedulerStatus, TaskStatus

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
def case_dask_executor() -> str:
    # HACK: This is a hack, see `scheduler` fixture
    return "dask"


@dataclass
class Dispatcher:
    """A simple class to dispatch functions to the scheduler by holding reference."""

    scheduler: Scheduler

    def __call__(self, f, *args, **kwargs) -> Callable[..., None]:
        """Dispatch a function to the scheduler."""

        def _f(*ignore_args, **ignore_kwargs):  # noqa: ARG001
            self.scheduler.dispatch(f, *args, **kwargs)

        return _f


def event_count(
    event: TaskStatus | SchedulerStatus,
    count: int,
) -> Callable[[Scheduler], bool]:
    def _f(scheduler: Scheduler) -> bool:
        return scheduler.counts[event] == count

    return _f


@fixture()
@parametrize_with_cases("executor", cases=".", has_tag="executor")
def scheduler(executor: Executor | str) -> Iterator[Scheduler]:
    # HACK: This is a hack, so we can yield and clean up
    if isinstance(executor, str) and executor == "dask":
        with LocalCluster(n_workers=2) as cluster, Client(cluster) as client:
            yield Scheduler(client.get_executor())
    elif isinstance(executor, Executor):
        yield Scheduler(executor)
    else:
        raise NotImplementedError(f"Executor {executor} not implemented")


def test_scheduler_timeout_stop_wait(scheduler: Scheduler) -> None:
    results: list[float] = []
    dispatcher = Dispatcher(scheduler)

    scheduler.on(scheduler.status.STARTED, dispatcher(sleep_and_return, 0.1))
    scheduler.on(scheduler.task.SUCCESS, results.append)

    end_status = scheduler.run(timeout=0.1, wait=True)

    assert end_status == scheduler.exitcode.TIMEOUT
    assert results == [0.1]
    assert scheduler.empty()


def test_scheduler_timeout_stop_no_wait(scheduler: Scheduler) -> None:
    results: list[float] = []
    dispatcher = Dispatcher(scheduler)

    scheduler.on(scheduler.task.SUCCESS, results.append)
    scheduler.on(scheduler.status.STARTED, dispatcher(sleep_and_return, 0.1))
    end_status = scheduler.run(timeout=0.01, wait=False)

    assert end_status == scheduler.exitcode.TIMEOUT
    assert results == []
    assert scheduler.empty()


def test_dispatch_within_callback(scheduler: Scheduler) -> None:
    results: list[float] = []
    dispatcher = Dispatcher(scheduler)

    scheduler.on(scheduler.task.SUCCESS, results.append)
    scheduler.on(scheduler.status.STARTED, dispatcher(sleep_and_return, 0.1))
    scheduler.on(
        scheduler.task.SUCCESS,
        dispatcher(sleep_and_return, 0.1),
        when=scheduler.count(scheduler.task.SUCCESS) < 2,
    )

    # Should run twice
    end_status = scheduler.run(timeout=1)

    assert end_status == scheduler.exitcode.EMPTY
    assert results == [0.1, 0.1]
    assert scheduler.empty()


def test_queue_empty_stop_criterion(scheduler: Scheduler) -> None:
    results: list[float] = []
    dispatcher = Dispatcher(scheduler)
    scheduler.on(scheduler.task.SUCCESS, results.append)

    scheduler.on(
        scheduler.status.STARTED,
        dispatcher(sleep_and_return, 0.1),
        dispatcher(sleep_and_return, 0.1),
        dispatcher(sleep_and_return, 0.1),
    )
    end_status = scheduler.run()

    assert end_status == scheduler.exitcode.EMPTY
    assert results == [0.1, 0.1, 0.1]
    assert scheduler.empty()


def test_stop_criterion(scheduler: Scheduler) -> None:
    results: list[float] = []
    dispatcher = Dispatcher(scheduler)

    scheduler.on(scheduler.status.STARTED, dispatcher(sleep_and_return, 0.1))
    scheduler.on(scheduler.task.SUCCESS, results.append)
    scheduler.on(scheduler.task.SUCCESS, dispatcher(sleep_and_return, 0.1))
    scheduler.on(
        scheduler.task.SUCCESS,
        scheduler.stop,
        when=scheduler.count(scheduler.task.SUCCESS) == 2,
    )
    end_status = scheduler.run(wait=True)

    assert end_status == scheduler.exitcode.STOPPED
    assert results == [0.1, 0.1, 0.1]
    assert scheduler.empty()


def test_error_handling(scheduler: Scheduler) -> None:
    results: list[float] = []
    errors: list[BaseException] = []
    dispatcher = Dispatcher(scheduler)

    def raise_error() -> None:
        raise ValueError("Error!")

    scheduler.on(scheduler.status.STARTED, dispatcher(sleep_and_return, 0.1))

    scheduler.on(scheduler.task.SUCCESS, results.append)
    scheduler.on(scheduler.task.ERROR, errors.append)

    scheduler.on(scheduler.task.ERROR, scheduler.stop)

    scheduler.on(
        scheduler.task.SUCCESS,
        dispatcher(sleep_and_return, 0.1),
        when=scheduler.count(scheduler.task.SUCCESS) < 2,
    )
    scheduler.on(
        scheduler.task.SUCCESS,
        dispatcher(raise_error),
        when=scheduler.count(scheduler.task.SUCCESS) == 2,
    )

    end_status = scheduler.run()

    assert end_status == scheduler.exitcode.STOPPED
    assert results == [0.1, 0.1]
    assert len(scheduler.queue) == 0
    assert len(errors) == 1
