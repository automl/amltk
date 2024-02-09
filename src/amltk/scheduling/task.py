"""A [`Task`][amltk.scheduling.task.Task] is a unit of work that can be scheduled by the
[`Scheduler`][amltk.scheduling.Scheduler].

It is defined by its `function=` to call. Whenever a `Task`
has its [`submit()`][amltk.scheduling.task.Task.submit] method called,
the function will be dispatched to run by a `Scheduler`.

When a task has returned, either successfully, or with an exception,
it will emit `@events` to indicate so. You can subscribe to these events
with callbacks and act accordingly.


??? example "`@events`"

    Check out the `@events` reference
    for more on how to customize these callbacks. You can also take a look
    at the API of [`on()`][amltk.scheduling.task.Task.on] for more information.

    === "`@on-result`"

        ::: amltk.scheduling.task.Task.on_result

    === "`@on-exception`"

        ::: amltk.scheduling.task.Task.on_exception

    === "`@on-done`"

        ::: amltk.scheduling.task.Task.on_done

    === "`@on-submitted`"

        ::: amltk.scheduling.task.Task.on_submitted

    === "`@on-cancelled`"

        ::: amltk.scheduling.task.Task.on_cancelled

??? tip "Usage"

    The usual way to create a task is with
    [`Scheduler.task()`][amltk.scheduling.scheduler.Scheduler.task],
    where you provide the `function=` to call.

    ```python exec="true" source="material-block" html="true"
    from amltk import Scheduler

    def f(x: int) -> int:
        return x * 2
    from amltk._doc import make_picklable; make_picklable(f)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(2)
    task = scheduler.task(f)

    @scheduler.on_start
    def on_start():
        task.submit(1)

    @task.on_result
    def on_result(future: Future[int], result: int):
        print(f"Task {future} returned {result}")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler)  # markdown-exec: hide
    ```

    If you'd like to simply just call the original function, without submitting it to
    the scheduler, you can always just call the task directly, i.e. `#!python task(1)`.

You can also provide [`Plugins`][amltk.scheduling.plugins.Plugin] to the task,
to modify tasks, add functionality and add new events.
"""
from __future__ import annotations

import logging
from asyncio import Future
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Concatenate, Generic, TypeVar
from typing_extensions import ParamSpec, Self, override

from dask.utils import funcname

from amltk._functional import callstring
from amltk._richutil.renderable import RichRenderable
from amltk.exceptions import SchedulerNotRunningError
from amltk.scheduling.events import Emitter, Event, Subscriber
from amltk.scheduling.plugins.plugin import Plugin

if TYPE_CHECKING:
    from rich.panel import Panel

    from amltk.scheduling.scheduler import Scheduler

logger = logging.getLogger(__name__)


P = ParamSpec("P")
P2 = ParamSpec("P2")

R = TypeVar("R")
R2 = TypeVar("R2")
CallableT = TypeVar("CallableT", bound=Callable)


class Task(Emitter, RichRenderable, Generic[P, R]):
    """The task class."""

    unique_ref: str
    """A unique reference to this task."""
    plugins: list[Plugin]
    """The plugins to use for this task."""
    function: Callable[P, R]
    """The function of this task"""
    scheduler: Scheduler
    """The scheduler that this task is registered with."""
    init_plugins: bool
    """Whether to initialize the plugins or not."""
    queue: list[Future[R]]
    """The queue of futures for this task."""
    emitter: Emitter
    """The emitter for events of this task."""

    on_submitted: Subscriber[Concatenate[Future[R], P], Any]
    """An event that is emitted when a future is submitted to the
    scheduler. It will pass the future as the first argument with args and
    kwargs following.

    This is done before any callbacks are attached to the future.
    ```python
    @task.on_submitted
    def on_submitted(future: Future[R], *args, **kwargs):
        print(f"Future {future} was submitted with {args=} and {kwargs=}")
    ```
    """
    on_done: Subscriber[[Future[R]], Any]
    """Called when a task is done running with a result or exception.
    ```python
    @task.on_done
    def on_done(future: Future[R]):
        print(f"Future {future} is done")
    ```
    """
    on_cancelled: Subscriber[[Future[R]], Any]
    """Called when a task is cancelled.
    ```python
    @task.on_cancelled
    def on_cancelled(future: Future[R]):
        print(f"Future {future} was cancelled")
    ```
    """
    on_result: Subscriber[[Future[R], R], Any]
    """Called when a task has successfully returned a value.
    Comes with Future
    ```python
    @task.on_result
    def on_result(future: Future[R], result: R):
        print(f"Future {future} returned {result}")
    ```
    """
    on_exception: Subscriber[[Future[R], BaseException], Any]
    """Called when a task failed to return anything but an exception.
    Comes with Future
    ```python
    @task.on_exception
    def on_exception(future: Future[R], error: BaseException):
        print(f"Future {future} exceptioned {error}")
    ```
    """

    SUBMITTED: Event[Concatenate[Future[R], P], Any] = Event("on-submitted")
    DONE: Event[[Future[R]], Any] = Event("on-done")

    CANCELLED: Event[[Future[R]], Any] = Event("on-cancelled")
    RESULT: Event[[Future[R], R], Any] = Event("on-result")
    EXCEPTION: Event[[Future[R], BaseException], Any] = Event("on-exception")

    def __init__(
        self: Self,
        function: Callable[P, R],
        scheduler: Scheduler,
        *,
        plugins: Plugin | Iterable[Plugin] = (),
        init_plugins: bool = True,
    ) -> None:
        """Initialize a task.

        Args:
            function: The function of this task
            scheduler: The scheduler that this task is registered with.
            plugins: The plugins to use for this task.
            init_plugins: Whether to initialize the plugins or not.
        """
        super().__init__(name=f"Task-{funcname(function)}")
        self.plugins: list[Plugin] = (
            [plugins] if isinstance(plugins, Plugin) else list(plugins)
        )
        self.function: Callable[P, R] = function
        self.scheduler: Scheduler = scheduler
        self.init_plugins: bool = init_plugins
        self.queue: list[Future[R]] = []

        # Set up subscription methods to events
        self.on_submitted = self.subscriber(self.SUBMITTED)
        self.on_done = self.subscriber(self.DONE)
        self.on_result = self.subscriber(self.RESULT)
        self.on_exception = self.subscriber(self.EXCEPTION)
        self.on_cancelled = self.subscriber(self.CANCELLED)

        if init_plugins:
            for plugin in self.plugins:
                plugin.attach_task(self)

    def futures(self) -> list[Future[R]]:
        """Get the futures for this task.

        Returns:
            A list of futures for this task.
        """
        return self.queue

    @property
    def n_running(self) -> int:
        """Get the number of futures for this task that are currently running."""
        return sum(1 for f in self.queue if not f.done())

    def running(self) -> bool:
        """Check if this task has any futures that are currently running."""
        return self.n_running > 0

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Future[R] | None:
        """Dispatch this task.

        Args:
            *args: The positional arguments to pass to the task.
            **kwargs: The keyword arguments to call the task with.

        Returns:
            The future of the task, or `None` if it was rejected for either reaching
            some max call limit or a plugin prevented if from being submitted.

        Raises:
            SchedulerNotRunningError: If the scheduler is not running.
                You can protect against this using,
                [`scheduler.running()`][amltk.scheduling.scheduler.Scheduler.running].
        """
        # Inform all plugins that the task is about to be called
        # They have chance to cancel submission based on their return
        # value.
        fn = self.function
        for plugin in self.plugins:
            items = plugin.pre_submit(fn, *args, **kwargs)
            if items is None:
                logger.debug(
                    f"Plugin '{plugin.name}' prevented {self} from being submitted"
                    f" with {callstring(self.function, *args, **kwargs)}",
                )
                return None

            fn, args, kwargs = items  # type: ignore

        try:
            future = self.scheduler.submit(fn, *args, **kwargs)
        except SchedulerNotRunningError as e:
            logger.exception("Scheduler is not running", exc_info=e)
            raise e
        except Exception as e:
            logger.exception("Error submitting task", exc_info=e)
            raise e

        self.queue.append(future)

        # We have the function wrapped in something will
        # attach tracebacks to errors, so we need to get the
        # original function name.
        msg = f"Submitted {callstring(self.function, *args, **kwargs)} from {self}."
        logger.debug(msg)

        self.on_submitted.emit(future, *args, **kwargs)

        # Process the task once it's completed
        # NOTE: If the task is done super quickly or in the sequential mode,
        # this will immediatly call `self._process_future`.
        future.add_done_callback(self._process_future)
        return future

    def _process_future(self, future: Future[R]) -> None:
        try:
            self.queue.remove(future)
        except ValueError as e:
            raise ValueError(f"{future=} not found in task queue {self.queue=}") from e

        if future.cancelled():
            self.on_cancelled.emit(future)
            return

        self.on_done.emit(future)

        exception = future.exception()
        if exception is not None:
            self.on_exception.emit(future, exception)
        else:
            result = future.result()
            self.on_result.emit(future, result)

    def attach_plugin(self, plugin: Plugin) -> None:
        """Attach a plugin to this task.

        Args:
            plugin: The plugin to attach.
        """
        self.plugins.append(plugin)
        plugin.attach_task(self)

    def _when_future_from_submission(
        self,
        future: Future[R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Access the future before the callbacks for the future are registered.

        This is primarly to allow subclasses of Task to know of the future obtained
        from submitting to the Scheduler, **before** the callbacks are registered.
        This can be required in the case the task finishes super quickly, meaning
        callbacks are registered before the subtask can do anything. This also
        necessarily happens in a sequential execution Scheduler.
        """

    @override
    def __repr__(self) -> str:
        kwargs = {"unique_ref": self.unique_ref, "plugins": self.plugins}
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"

    @override
    def __rich__(self) -> Panel:
        from rich.console import Group as RichGroup
        from rich.panel import Panel
        from rich.text import Text
        from rich.tree import Tree

        from amltk._richutil import Function

        items: list[RichRenderable | Tree] = []

        if any(self.plugins):
            for plugin in self.plugins:
                items.append(plugin)

        tree = Tree(label="", hide_root=True)
        tree.add(self)
        items.append(tree)

        return Panel(
            RichGroup(*items),
            title=Function(self.function, prefix="Task").__rich__(),
            title_align="left",
            border_style="deep_sky_blue2",
            subtitle=Text("Ref: ").append(self.unique_ref, "yellow italic"),
        )
