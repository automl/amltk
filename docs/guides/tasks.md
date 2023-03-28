AutoML-toolkit was designed to make offloading computation
away from the main process __easy__, to foster increased ability for
interactability, deployment and control. At the same time,
we wanted to have an event based system to manage the complexity
that comes with AutoML systems, all while making the API intuitive
and extensible.

By the end of this guide, we hope that the following code, it's options
and it's inner working becomes easy to understand.

```python
from byop.scheduling import Scheduler, Task

def the_anwser(a: int, b: int, c: int) -> int:
    the_answer = ...
    return the_answer

scheduler = Scheduler(executor=...)
task = Task(compute_42, scheduler, ...)

@scheduler.on_start
def start_computing() -> None:
    task(1, 2, 3)

@task.on_returned
def print_answer_found(the_answer: int) -> None:
    print(f"{the_answer} was found!")

@task.on_exception
def print_exception(exception: BaseException) -> None:
    print(f"Finding the answer failed: {exception}")

scheduler.run()
```


This guide starts with a simple introduce to `amltk`'s event system, which
act as the gears through which the whole system moves.
After that, we introduce the engine, the [`Scheduler`][byop.scheduling.Scheduler]
and how this interacts with python's built in interface [`Executor`][concurrent.futures.Executor]
to offload compute to processes, compute nodes or even cloud resources.
However the `Scheduler` is rather useless without some fuel.
For this we present [`Tasks`][byop.scheduling.Task], the compute to actually
perform with the `Scheduler` and start the system's gears turning.


## Events
At the core of of AutoML-Toolkit is an [`EventManager`][byop.events.EventManager]
whose sole purpose is to allow you to do two things:

* [`emit`][byop.events.EventManager.emit]: Emit some event with arguments.
* [`on`][byop.events.EventManager.on]: Register a function which gets called when a certain
  event is emitted.

![Event Manager](../images/scheduler-guide-events.svg)

There is some _sugar_ the `EventManager` provides but we will introduce
them as we go along.

## Scheduler
The engine of AutoML-Toolkit is the [`Scheduler`][byop.scheduling.Scheduler].
It requires one thing to function and that's an
[`Executor`][concurrent.futures.Executor], an interface defined
by core python for something that takes a function `f` and
it's arguments, returning a [`Future`][concurrent.futures.Future].

![Scheduler Events](../images/scheduler-guide-first.svg)

!!! Note "A [`Future`][concurrent.futures.Future]"

    This is an object which will have the result or exception of some
    compute in the future.

Using just these, we can start to scaffold an entire event based framework which
starts from the `Scheduler` and builds outwards to the cleaner abstraction, a [`Task`][byop.scheduling.Task].

### Creating a Scheduler
We can create a simple `Scheduler` that uses the
builtin [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] to
run compute in a seperate process on the same machine.

=== "Easy"

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler.with_processes(2)
    ```

=== "Explicit"

    ```python
    from concurrent.futures import ProcessPoolExecutor
    from byop.scheduling import Scheduler

    executor = ProcessPoolExecutor(max_workers=2)
    scheduler = Scheduler(executor=executor)
    ```

Python also defines a [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor]
but there are some known drawbacks to offloading heavy compute to threads.

Some popular distributions frameworks which support the `Executor` interface:

* [`Dask`](https://distributed.dask.org/en/stable/): The `Client` from dask
  has a method [`get_executor()`](https://distributed.dask.org/en/stable/api.html?highlight=get_executor#distributed.Client.get_executor)
  which exposes the `Client` following the `Executor` interface.
* [`dask-jobqueue`](https://jobqueue.dask.org/en/latest/): Provides `Executor`
  interfaces that support workers on compute nodes for some common clusters.
* We plan to support more integrations, for example [Apache Airflow](https://airflow.apache.org).
  If there's an executor background you wish to integrate, we would be happy
  to consider it and greatly appreciate a PR!

### Listening to Scheduler Events with `on_start()`
!!! info inline end "Events"

    The `Scheduler` defines even more events which we can subscribe to:

    * [`EMPTY`][byop.scheduling.Scheduler.EMPTY]: An event to signal
      that there is nothing currently running in the scheduler. Subscribe
      with `on_empty`.
    * [`TIMEOUT`][byop.scheduling.Scheduler.TIMEOUT]: An event to signal
      the scheduler has reached its time limit. Subscribe with `on_timeout()`.
    * [`STOP`][byop.scheduling.Scheduler.STOP]: An event to signal the
      scheduler has been stopped explicitly with
      [`scheduler.stop()`][byop.scheduling.Scheduler.stop]. This can
      be subscribed to with `on_stop()`.

    In general, each event can be listened to with `on_event_name` where
    `event_name()` is the lowercase version of the `Event`.

    The complete list of events for [`Scheduler`][byop.scheduling.Scheduler].


The `Scheduler` defines many events which are emitted depending on its state.
One example of this is [`STARTED`][byop.scheduling.Scheduler.STARTED], an
event to signal the scheduler has started and ready to accept tasks.
We can subscribe to this event and trigger our callback with `on_start()`.
We provide two ways to do so, one with _decorators_ and another in a _functional_
way.

=== "Decorators"

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    @scheduler.on_start  # (1)!
    def print_hello() -> None:
        print("hello")
    ```

    1. You can decorate your callback and it will be called when the
    scheduler emits the [`STARTED`][byop.scheduling.Scheduler.STARTED] event.

=== "Functional"

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    def print_hello() -> None:
        print("hello")

    scheduler.on_start(print_hello)  # (1)!
    ```

    1. You can just pass in your callback and it will be called when the
    scheduler emits the [`STARTED`][byop.scheduling.Scheduler.STARTED] event.

It's worth noting that even though we are using an event based system, we
are still guaranteed deterministic execution of the callbacks for any given
event. The source of indeterminism is the order in which the events are
emitted, as discussed later in the [tasks section](#tasks).

We can access all the counts of all events through the
[`scheduler.counts`][byop.scheduling.Scheduler.counts] property.
This is a `dict` which has the events as keys and the amount of times
it was emitted as the values.

??? Note "Unnecessary Details"

    While not necessary to know, `on_start` is actually a callable object
    called a [`Subscriber`][byop.events.Subscriber] which takes a variety
    of arguments you can find [here][byop.events.Subscriber.__call__].

    * `name`: The name to give the callback. Used for reference in the logs
    and in the [`EventManager`][byop.events.EventManager].
    * `when`: A predicate which takes no arguments and returns True or
    False on whether this callback should be called when the event is emitted.
    * `limit`: The amount of times this callback can be called.
    * `repeat`: How many times this callback will be called when the event
    has been emitted.
    * `every`: An int specifying the callback will be called every `every`
    times the event is emitted.


### How to `run()` and `stop()` the Scheduler
We can run the scheduler in two different ways, synchronously with
[`run()`][byop.scheduling.Scheduler.run]
which is blocking, or with [`async_run()`][byop.scheduling.Scheduler.async_run]
which runs the scheduler in an [`asyncio`][asyncio] loop, useful for
interactive apps or servers.

!!! Note inline end "Arguments to `run()`"

    * `timeout`: The scheduler will automatically stop once `timeout` many seconds
    is reached.
    * `end_on_empty`: The scheduler will stop once there is no pending compute
    running. The scheduler is out of fuel and no more events can occur.
    * `wait`: Whether the scheduler should wait for currently running compute
    to finish. Check out the `terminate` argument to
    [`Scheduler`][byop.scheduling.Scheduler] for finer grained control on the
    limitations of setting this to `False`.

    Check it's documentation [`run()`][byop.scheduling.Scheduler.run]

Once the scheduler finishes, it will return an [`ExitCode`][byop.scheduling.Scheduler.ExitCode], indicating why
the scheduler finished.

```python hl_lines="5"
from byop.scheduling import Scheduler

scheduler = Scheduler(...)

exit_code = scheduler.run()
```


Lastly, we can always forcibly stop the scheduler with
[`stop()`][byop.scheduling.Scheduler.stop], whether it be in a callback
or elsewhere.

```python hl_lines="7"
from byop.scheduling import Scheduler

scheduler = Scheduler(...)

@scheduler.on_start
def stop_immediatly() -> None:
    scheduler.stop()

scheduler.run()
```

### Injecting fuel with `submit()`
One thing that's missing from all of this is actually running something
with the scheduler. We first introduce the manual way to do so with
[`submit()`][byop.scheduling.Scheduler.submit] and follow up with
the final part of this guide by introducing [`Tasks`][byop.scheduling.Task],
supercharging functions with events.

```python hl_lines="18"
import time
import random

from asnycio import Future
from byop.scheduling import Scheduler

def long_running_function(x: int) -> int:
    time.sleep(42)
    return x

futures: list[Future[int]] = []  # (3)!

scheduler = Scheduler(...)

@scheduler.on_start(repeat=3)  # (1)!
def print_result() -> None:
    x = random.randrange(10)
    future = scheduler.submit(long_running_function, x)  # (2)!
    futures.append(future)

scheduler.run()

results = [future.result() for future in futures]
```

1. Using the `repeat` argument, we can call this callback 3 times.
  once the scheduler starts
2. We submit a compute function to the scheduler, which will run it
  in the background. Thanks to typing `submit()` will complain if
  you pass arguments that don't match the function signature.
3. We keep track of the futures returned by `submit()` so we can
  access the results later.

Notice that working with `Futures` is not very convenient and this sample above
does not start to account for cancelled of failed futures. We also have no way
to actually process a `Future` while the scheduler is running which is very inconvenient.

[`Tasks`][byop.scheduling.Task] can help us with this.

## Tasks
[`Tasks`][byop.scheduling.Task] are a powerful way to run compute functions
with the scheduler. In their simplest form, they are wrapped functions
which when called, are sent off to the work to be computed. In practice,
they hook your functions into the event system, provide methods to limit
resource consumption and provide a convenient way to express special kinds
of tasks, with the possibility to create their own custom events to hook
into.

!!! Note "Provided Extensions of `Task`"

    * [`CommTask`][byop.scheduling.CommTask]s are tasks which are given
    a [`Comm`][byop.scheduling.Comm] as their first argument
    for two-way communicaiton with the main process.
    * [`Trial.Task`][byop.optimization.Trial.Task]s are tasks particular to
    optimization in AutoML-toolkit, which are given an optimization [`Trial`][byop.optimization.Trial]
    as their first argument and should return a [`Trial.Report`][byop.optimization.Trial.Report]
    for reporting how the optimization run went.

### Using a `Task`
To create a task, we need to wrap a function, let's use the same example from above
but with tasks.

=== "Using Tasks"

    ```python hl_lines="14"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    results: list[int] = []
    scheduler = Scheduler(...)

    task = Task(long_running_function, scheduler)  # (1)!

    @scheduler.on_start(repeat=3)
    def print_result() -> None:
        x = random.randrange(10)
        task(x) # (2)!

    @task.on_returned
    def save_result(result: int) -> None:
        results.append(result)

    @task.on_exception
    def stop_scheduler(exception: BaseException) -> None:
        print(exception)
        scheduler.stop()

    scheduler.run()
    ```

    1. To type this properly, it's `Task[[int], int]`, similar
       to how you would type a [`Callable`][typing.Callable].
    2. To run the task, we simply call it with our arguments.
       We can be type safe and `mypy` will give you an error
       if your arguments do not match the function signature.

=== "Using Futures Directly"

    ```python
    import time
    import random

    from asnycio import Future
    from byop.scheduling import Scheduler

    def long_running_function(x: int) -> int:
        time.sleep(42)
        return x

    results: list[int] = []
    scheduler = Scheduler(...)

    futures: list[Future[int]] = []

    @scheduler.on_start(repeat=3)
    def print_result() -> None:
        x = random.randrange(10)
        future = scheduler.submit(long_running_function, x)
        futures.append(future)

    scheduler.run()

    for future in futures:
      exception = future.exception()
      if exception is not None:
          print(exception)
      else:
          result = future.result()
          results.append(result)
    ```

If you compare the two examples, you'll see that handling futures is
abstracted away. Moreover, your callbacks will be handled as soon as a `Task`
completes, something not possible using just raw `Future`s alone.

Notice that we pass in the `scheduler` as the second argument to the `Task`.
This is because the `Task` needs to know which scheduler to use to submit
the compute function to.

Subscribing to events on the task is done in the exact same
manner as the scheduler, e.g. `on_returned()` and `on_exception()`.

=== "Decorators"

    ```python
    from byop.scheduling import Task

    task = Task(...)

    @task.on_returned
    def do_something_with_result(result) -> None:
        ...
    ```

=== "Functional"

    ```python
    from byop.scheduling import Task

    task = Task(...)

    def do_something_with_result(result) -> None:
        ...

    task.on_returned(do_something_with_result)
    ```

It's worth noting here that for any given event the order in which callbacks
are called is guaranteed to be the same as the order in which they were
registered. However the order of events is not guaranteed, meaning that
the list of results may change each time this code sample is run, depending
on how long the sleep lasts.

### Limiting resource consumption
Tasks can be used to limit the amount of resources a function can use.
This is done using the following arguments to `Task`:

=== "Call Limit"

    The maximum number of times this task can be run.
    If this limit is reached and the task is called again, the
    [`CALL_LIMIT_REACHED`][byop.scheduling.Task.CALL_LIMIT_REACHED] event will be emitted
    and the task will not be executed.
    This can be subscribed to with
    `on_call_limit_reached()`

    ```python
    from byop.scheduling import Task

    task = Task(..., call_limit=5)

    @task.on_call_limit_reached
    def print_it(*args, **kwargs) -> None:
        print(f"Task was already called {task.n_called} times")
        print(f"Failed to run {task=} with {args=} and {kwargs=}")
    ```

=== "Concurrency"

    The maximum number of concurrent executions of this task.
    If the concurrent limit is reached and the task is called again, the
    [`CONCURRENT_LIMIT_REACHED`][byop.scheduling.Task.CONCURRENT_LIMIT_REACHED] event will be emitted
    and the task will not be executed.
    This can be subscribed to with
    `on_concurrent_limit_reached()`.

    ```python
    from byop.scheduling import Task

    task = Task(..., concurrent_limit=3)

    @task.on_concurrent_limit_reached
    def print_it(*args, **kwargs) -> None:
        print(f"Task already running {self.n_running} workers")
        print(f"Failed to run {task=} with {args=} and {kwargs=}")
    ```

=== "Memory"

    The maximum amount of memory this task can use.
    If the memory limit triggered and the function crashes as a result,
    the [`MEMORY_LIMIT_REACHED`][byop.scheduling.Task.MEMORY_LIMIT_REACHED] event will be emitted.
    This can be subscribed to with
    `on_memory_limit_reached()`.

    ```python
    from byop.scheduling import Task

    task = Task(..., memory_limit=(2, "gb")) # (1)!

    @task.on_memory_limit_reached
    def print_it(future: Future, exception: BaseException) -> None:
        print(f"Task reached memory limit, {future=} failed.")
        print(f"Failed with {exception=}")
    ```

    1. Check out the parameters on how to set the memory limit
    [here](https://github.com/automl/pynisher#parameters)

    !!! Note "Memory Limits with Pynisher"

        Pynisher has some limitations with memory on Mac and Windows:
        https://github.com/automl/pynisher#features

=== "CPU time"

    The maximum amount of CPU time this task can use.
    If the CPU time limit triggered and the function crashes as a result,
    the [`TIMEOUT`][byop.scheduling.Task.TIMEOUT] and
    [`CPU_TIME_LIMIT_REACHED`][byop.scheduling.Task.CPU_TIME_LIMIT_REACHED] events will be emitted.
    This can be subscribed to with `on_timeout()` and
    `on_cpu_time_limit_reached()` respectively.

    ```python
    from byop.scheduling import Task

    task = Task(..., cpu_time_limit=(60, "s"))

    @task.on_cpu_limit_reached
    def print_it(future: Future, exception: BaseException) -> None:
        print(f"Task reached memory limit, {future=} failed.")
        print(f"Failed with {exception=}")
    ```

    1. Check out the parameters on how to set the cpu time limit
    [here](https://github.com/automl/pynisher#parameters)

    !!! Note "CPU Time Limits with Pynisher"

        Pynisher has some limitations with cpu timing on Mac and Windows:
        https://github.com/automl/pynisher#features

=== "Wall time"

    The maximum amount of wall clock time this task can use.
    If the wall clock time limit triggered and the function crashes as a result,
    the [`TIMEOUT`][byop.scheduling.Task.TIMEOUT] and
    [`WALL_TIME_LIMIT_REACHED`][byop.scheduling.Task.WALL_TIME_LIMIT_REACHED] events will be emitted.
    This can be subscribed to with `on_timeout()` and
    `on_wall_time_limit_reached()` respectively.

    ```python
    from byop.scheduling import Task

    task = Task(..., wall_time_limit=(5, "m"))  # (1)!

    @task.on_walltime_limit_reached
    def print_it(future: Future, exception: BaseException) -> None:
        print(f"Task reached memory limit, {future=} failed.")
        print(f"Failed with {exception=}")
    ```

    1. Check out the parameters on how to set the wall time limit
    [here](https://github.com/automl/pynisher#parameters)
