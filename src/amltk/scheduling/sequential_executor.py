"""A concurrent.futures.Executor interface that forces sequential execution."""
from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

R = TypeVar("R")
P = ParamSpec("P")


class SequentialExecutor(Executor):
    """A [Executor][concurrent.futures.Executor] interface for sequential execution."""

    def submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        """Submit a function to be executed.

        Args:
            fn: The function to execute.
            *args: The positional arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            A future that is already resolved with the result/exception of the function.
        """
        future: Future[R] = Future()
        future.set_running_or_notify_cancel()

        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)

        return future
