AutoML-toolkit was designed to make offloading computation
away from the main process __easy__, to foster increased ability for
interactability, deployment and control. At the same time,
we wanted to have an event based system to manage the complexity
that comes with AutoML systems, all while making the API intuitive
and extensible. This allows workflows which encourage gradual development
and experimentation, where the user can incrementally add more functionality
and complexity to their system.

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

=== "Diagram"

    ![Event Manager](../images/scheduler-guide-events.svg)

=== "Code"


    ```python
    from byop.events import EventManager

    def f(a: int, b: str, c: float) -> None:
        ...

    def g(a: int, b: str, c: float) -> None:
        ...

    def h(a: int, b: str, c: float) -> None:
        ...

    event_manager = EventManager(name="manager-name")

    # Subscribe the callbacks to the "hello" event
    event_manager.on("hello", f)
    event_manager.on("hello", g)
    event_manager.on("hello", h)

    # ... some time later

    # Will call all functions subscribed to the hello event
    event = "hello"
    event_manager.emit(event, 10, "world", 3.14)
    ```

We can technically define any [`Hashable`][typing.Hashable] object as an event, such as `#!python "hello"`,
but typically we create an [`Event`][byop.events.Event] object. By themselves they don't add much
but when using python's
typing and mypy/pyright, this let's us make sure the callbacks
registered to an event are compatible with the arguments passed to
[`emit`][byop.events.EventManager.emit].

=== "Creating an Event"

    ```python hl_lines="11"
    from byop.events import EventManager, Event

    def f(a: int, b: str, c: float) -> None:
        ...

    def h(x: str) -> None:
        ...

    event_manager = EventManager(name="manager-name")

    event = Event(name="hello")

    # Subscribe the callbacks to the "hello" event
    event_manager.on(event, f)
    event_manager.on(event, h)  # <--- This is not compatible with the emit below

    # ... some time later
    event_manager.emit(event, 10, "world", 3.14)  # <--- Exception happens when `h` is called
    ```

=== "Typed version"

    ```python hl_lines="11"
    from byop.events import EventManager, Event

    def f(a: int, b: str, c: float) -> None:
        ...

    def h(x: str) -> None:
        ...

    event_manager = EventManager(name="manager-name")

    event: Event[int, str, float] = Event(name="hello")

    # Subscribe the callbacks to the "hello" event
    event_manager.on(event, f)
    event_manager.on(event, h)  # <--- Mypy will tell you there's an error here

    # ... some time later
    event_manager.emit(event, 10, "world", 3.14)
    ```

There is some _sugar_ the `EventManager` provides but we will introduce
them as we go along.

## Scheduler
The engine of AutoML-Toolkit is the [`Scheduler`][byop.scheduling.Scheduler].
It requires one thing to function and that's an
[`Executor`][concurrent.futures.Executor], an interface defined
by core python for something that takes a function `f` and
it's arguments, returning a [`Future`][concurrent.futures.Future].

![Scheduler Events](../images/scheduler-guide-first.svg)

!!! quote "A [`Future`][concurrent.futures.Future]"

    This is a builtin Python object which will have the result or exception of some
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


Some popular distributions frameworks which support the `Executor` interface:

-   :material-language-python: **Python**

    Python supports the `Executor` interface natively with the
    [`concurrent.futures`][concurrent.futures] module for processes with the
    [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] and
    [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] for threads.

    ??? example

        === "Process Pool Executor"

            ```python
            from concurrent.futures import ProcessPoolExecutor
            from byop.scheduling import Scheduler

            executor = ProcessPoolExecutor(max_workers=2)
            scheduler = Scheduler(executor=executor)
            ```

        === "Thread Pool Executor"

            ```python
            from concurrent.futures import ThreadPoolExecutor
            from byop.scheduling import Scheduler

            executor = ThreadPoolExecutor(max_workers=2)
            scheduler = Scheduler(executor=executor)
            ```

            !!! danger "Why to not use threads"

                Python also defines a [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor]
                but there are some known drawbacks to offloading heavy compute to threads. Notably,
                there's no way in python to terminate a thread from the outside while it's running.
---

-   :simple-dask: [`dask`](https://distributed.dask.org/en/stable/)

    Dask and the supporting extension [`dask.distributed`](https://distributed.dask.org/en/stable/)
    provide a robust and flexible framework for scheduling compute across workers.

    ??? example

        ```python hl_lines="5"
        from dask.distributed import Client
        from byop.scheduling import Scheduler

        client = Client(...)
        executor = client.get_executor()
        scheduler = Scheduler(executor=executor)
        ```
---

-   :simple-dask: [`dask-jobqueue`](https://jobqueue.dask.org/en/latest/)

    A package for scheduling jobs across common clusters setups such as
    PBS, Slurm, MOAB, SGE, LSF, and HTCondor.

    ??? example

        ```python hl_lines="16"
        from dask_jobqueue import SLURMCluster
        from byop.scheduling import Scheduler

        n_workers = 256

        cluster = SLURMCluster(
            memory="2GB",  # Memory per job
            processes=1,  # Processes per job
            cores=1,  # Cores per job
            job_extra_directives=["--time 0-00:10:00"],  # Duration
            queue="partition-name",
        )

        cluster.adapt(maximum_jobs=256)  # Automatically scale up to 256 jobs

        executor = cluster.get_client().get_executor()
        scheduler = Scheduler(executor=executor)
        ```

        !!! info "Modifying the launch command"

            On some cluster commands, you'll need to modify the launch command.
            You can use the following to do so:

            ```python
            SLURMCluster.job_cls.submit_command = "sbatch <extra-arguments>"
            ```

---

-   :simple-ray: [`ray`](https://docs.ray.io/en/master/)

    Ray is an open-source unified compute framework that makes it easy
    to scale AI and Python workloads
    — from reinforcement learning to deep learning to tuning,
    and model serving.

    ??? info "In progress"

        Ray is currently in the works of supporting the Python
        `Executor` interface. See this [PR](https://github.com/ray-project/ray/pull/30826)
        for more info.

---

-   :simple-apacheairflow: [`airflow`](https://airflow.apache.org/)

    Airflow is a platform created by the community to programmatically author,
    schedule and monitor workflows. Their list of integrations to platforms is endless
    but features compute platforms such as Kubernetes, AWS, Microsoft Azure and
    GCP.

    ??? info "Planned"

        We plan to support `airflow` in the future. If you'd like to help
        out, please reach out to us!

---

If there's an executor background you wish to integrate, we would be happy
to consider it and greatly appreciate a PR!

### Subscribing to Scheduler Events

The `Scheduler` defines many events which are emitted depending on its state.
One example of this is [`STARTED`][byop.scheduling.Scheduler.STARTED], an
event to signal the scheduler has started and ready to accept tasks.
We can subscribe to this event and trigger our callback with `on_start()`.

!!! info inline end "Events"

    The `Scheduler` defines even more events which we can subscribe to:

    ---
    [`STARTED`][byop.scheduling.Scheduler.STARTED] : `on_start()`.

    An event to signal that the scheduler is now up and running.

    ---

    [`FINISHING`][byop.scheduling.Scheduler.FINISHING] : `on_finishing()`.

    An event to signal the scheduler is still running but is waiting for
    currently running tasks to finish.

    ---

    [`FINISHED`][byop.scheduling.Scheduler.FINISHING] : `on_finished()`.

    An event to signal the scheduler is no longer running and this is
    the last event it will emit.

    ---

    [`EMPTY`][byop.scheduling.Scheduler.EMPTY] : `on_empty()`.

    An event to signal that there is nothing currently running in the scheduler.

    ---

    [`TIMEOUT`][byop.scheduling.Scheduler.TIMEOUT] : `on_timeout()`.

    An event to signal the scheduler has reached its time limit.

    ---

    [`STOP`][byop.scheduling.Scheduler.STOP] : `on_stop()`.

    An event to signal the scheduler has been stopped explicitly with
    [`scheduler.stop()`][byop.scheduling.Scheduler.stop].

    ---

    In general, each event can be listened to with `on_event_name` where
    `event_name()` is the lowercase version of the `Event`.

    The complete list of events for [`Scheduler`][byop.scheduling.Scheduler].

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

There's a variety of parameters you can use to customize the behavior of
the callback. You can find the full list of parameters [here][byop.events.Subscriber.__call__].

=== "`name=`"

    Register a custom name with the callback to use for logging
    or to [`remove`][byop.events.Subscriber.remove] the callback later.

    ```python hl_lines="5"
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    @scheduler.on_start(name="my_callback")
    def print_hello() -> None:
        print("hello")
    ```

=== "`when=`"

    A callable which takes no arguments and returns a `bool`. The callback
    will only be called when the `when` callable returns `True`.

    ```python hl_lines="10"
    import random
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    def random_bool() -> bool:
        return random.choice([True, False])

    # Randomly decide whether to call the callback
    @scheduler.on_start(when=random_bool)
    def print_hello() -> None:
        print("hello")
    ```
=== "`limit=`"

    Limit the number of times a callback can be called, after which, the callback
    will be ignored.

    ```python hl_lines="6"
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    # Make sure it can only be called once
    @scheduler.on_start(limit=1)
    def print_hello() -> None:
        print("hello")
    ```

=== "`repeat=`"

    Repeat the callback a certain number of times, every time the event is emitted.

    ```python hl_lines="6"
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    # Print "hello" 3 times when the scheduler starts
    @scheduler.on_start(repeat=3)
    def print_hello() -> None:
        print("hello")
    ```

=== "`every=`"

    Only call the callback every `every` times the event is emitted. This
    includes the first time it's called.

    ```python hl_lines="6"
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    # Print "hello" only every 3 times the scheduler starts.
    @scheduler.on_start(every=3)
    def print_hello() -> None:
        print("hello")
    ```

It's worth noting that even though we are using an event based system, we
are still guaranteed deterministic execution of the callbacks for any given
event. The source of indeterminism is the order in which the events are
emitted, as discussed later in the [tasks section](#tasks).

We can access all the counts of all events through the
[`scheduler.counts`][byop.scheduling.Scheduler.counts] property.
This is a `dict` which has the events as keys and the amount of times
it was emitted as the values.

??? info "Unnecessary Details"

    While not necessary to know, `on_start` is actually a callable object
    called a [`Subscriber`][byop.events.Subscriber].

    You can create one of these quite easily!

    ```python hl_lines="3 6"
    from byop.events import EventManager, Event

    USER_JOINED: Event[str] = Event("user-joined")  # (1)!

    event_manager = EventManager(name="event_manager")
    on_user_joined = event_manager.subscriber(USER_JOINED)

    @on_user_joined
    def print_hello(username: str) -> None:
        print(f"hello {username}, welcome!")

    on_user_joined(print_hello)
    ```

    1. We define our event with a type parameter to specify the type of
    data we will pass to the callback.


### Running and Stopping the Scheduler
We can run the scheduler in two different ways, synchronously with
[`run()`][byop.scheduling.Scheduler.run]
which is blocking, or with [`async_run()`][byop.scheduling.Scheduler.async_run]
which runs the scheduler in an [`asyncio`][asyncio] loop, useful for
interactive apps or servers.

Once the scheduler finishes, it will return an [`ExitCode`][byop.scheduling.Scheduler.ExitCode], indicating why
the scheduler finished.

=== "`run()`"

    Default behavior is to run the scheduler until it's out of things
    to run as there will be no more events to fire. In this case
    it will return [`ExitCode.EXHAUSTED`][byop.scheduling.Scheduler.ExitCode.EXHAUSTED].

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    exit_code = scheduler.run()
    ```

=== "`stop()`"

    We can always forcibly stop the scheduler with
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


=== "`run(timeout=...)`"

    You can set a timeout for the scheduler which means it will shutdown after
    `timeout=` seconds. We explicitly pass `end_on_empty=False` here to prevent
    the scheduler from shutting down due to being out of fuel. In this
    case the scheduler will return
    [`ExitCode.TIMEOUT`][byop.scheduling.Scheduler.ExitCode.TIMEOUT].

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    exit_code = scheduler.run(timeout=10, end_on_empty=False)
    ```

=== "`run(wait=...)`"

    We can also set `wait=False` to prevent the scheduler from waiting for
    currently running compute to finish if it stopped for whatever reason.
    In this case the scheduler will return attempt to shutdown the executor
    but you can pass in `terminate` to
    [`Scheduler`][byop.scheduling.Scheduler] to define how this is done.

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    scheduler.run(wait=False)
    ```

=== "`async_run()`"

    For running applications such as servers or interactive apps, we can use
    [`async_run()`][byop.scheduling.Scheduler.async_run] which runs the scheduler
    in an [`asyncio`][asyncio] loop. This is useful for running the scheduler
    in the background while you do other things, such as respond to incoming
    HTTP requests or draw and manage UI components. This takes the same arguments
    as [`run()`][byop.scheduling.Scheduler.run].

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler(...)

    # ... somewhere in your async code
    await scheduler.async_run()
    ```

---

### Submitting Compute to the Scheduler
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
    if x == 0:
        raise ValueError("x cannot be 0")

    time.sleep(42)
    return x

futures: list[Future[int]] = []  # (3)!

scheduler = Scheduler(...)

@scheduler.on_start(repeat=3)  # (1)!
def submit_task() -> None:
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

!!! info "Provided Extensions of `Task`"

    * [`CommTask`][byop.scheduling.CommTask]s are tasks which are given
    a [`Comm`][byop.scheduling.Comm] as their first argument
    for two-way communicaiton with the main process.
    * [`Trial.Task`][byop.optimization.Trial.Task]s are tasks particular to
    optimization in AutoML-toolkit, which are given an optimization [`Trial`][byop.optimization.Trial]
    as their first argument and should return a [`Trial.Report`][byop.optimization.Trial.Report]
    for reporting how the optimization run went.

### Creating a Task
To create a task, we simply wrap the function we will want to distribute and pass
in the scheduler we will use for managing the distribution.

```python
import time
import random

from byop.scheduling import Scheduler, Task

def long_running_function(x: int) -> int:
    if x == 0:
        raise ValueError("x cannot be 0")

    sleep_time = random.randrange(5)
    time.sleep(sleep_time)
    return x

scheduler = Scheduler(...)
task = Task(long_running_function, scheduler)
```

### Using a Task with Incremental Development

To submit a task, we simply call it like a normal function. We can
pass in any arguments that match the function signature. The call
to the task will return a [`Future`][asyncio.Future] we can use if
we like but the proffered way to handle the results is to use the
`on_returned` and `on_exception` events.

Here we demonstrate how we can gradually build up complexity to
test our system and add functionality with small incremental changes

=== "Initial Test"

    We start by just having a simple test of submitting the task once
    when the scheduler starts and printing the outcome.

    ```python hl_lines="17 18 19 20 22 23 24 26 27 28"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        if x == 0:
            raise ValueError("x cannot be 0")

        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    scheduler = Scheduler(...)
    task = Task(long_running_function, scheduler)

    @scheduler.on_start
    def submit_task() -> None:
        x = random.randrange(10)
        task(x)

    @task.on_returned
    def print_result(result: int) -> None:
        print(result)

    @task.on_exception
    def print_exception(exception: BaseException) -> None:
        print(exception)

    scheduler.run()
    ```

=== "Save results"

    We realise we actually need to store the results somewhere so
    we can add new functionality without disturbing our previous code.

    ```python hl_lines="26 27 28 29 30"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        if x == 0:
            raise ValueError("x cannot be 0")

        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    scheduler = Scheduler(...)
    task = Task(long_running_function, scheduler)

    @scheduler.on_start
    def submit_task() -> None:
        x = random.randrange(10)
        task(x)

    @task.on_returned
    def print_result(result: int) -> None:
        print(result)

    results: list[int] = []

    @task.on_returned
    def save_result(result: int) -> None:
        results.append(result)

    @task.on_exception
    def print_exception(exception: BaseException) -> None:
        print(exception)

    scheduler.run()
    ```

=== "Concurrency"

    We now see if we can submit our task 3 times at the same time
    once the scheduler starts.

    ```python hl_lines="17"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        if x == 0:
            raise ValueError("x cannot be 0")

        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    scheduler = Scheduler(...)
    task = Task(long_running_function, scheduler)

    @scheduler.on_start(repeat=3)
    def submit_task() -> None:
        x = random.randrange(10)
        task(x)

    @task.on_returned
    def print_result(result: int) -> None:
        print(result)

    results: list[int] = []

    @task.on_returned
    def save_result(result: int) -> None:
        results.append(result)

    @task.on_exception
    def print_exception(exception: BaseException) -> None:
        print(exception)

    scheduler.run()
    ```

=== "Repeat for a time"

    Now that we have a working system, we want to keep it running
    for a while and keep the workers busy. We can do this by using
    just resubmitting the task once one of it's submissions is done,
    regardless of a result or exception.

    ```python hl_lines="18 37"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        if x == 0:
            raise ValueError("x cannot be 0")

        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    scheduler = Scheduler(...)
    task = Task(long_running_function, scheduler)

    @scheduler.on_start(repeat=3)
    @task.on_done
    def submit_task() -> None:
        x = random.randrange(10)
        task(x)

    @task.on_returned
    def print_result(result: int) -> None:
        print(result)

    results: list[int] = []

    @task.on_returned
    def save_result(result: int) -> None:
        results.append(result)

    @task.on_exception
    def print_exception(exception: BaseException) -> None:
        print(exception)

    scheduler.run(timeout=60)
    ```

=== "Stopping Criteria"

    Lastly, we realise we want to stop the scheduler once we have 50 results
    or an exception occured. We can add in these stopping criterion quite fluidly.

    ```python hl_lines="37 38 39 41 42 43"
    import time
    import random

    from byop.scheduling import Scheduler, Task

    def long_running_function(x: int) -> int:
        if x == 0:
            raise ValueError("x cannot be 0")

        sleep_time = random.randrange(5)
        time.sleep(sleep_time)
        return x

    scheduler = Scheduler(...)
    task = Task(long_running_function, scheduler)

    @scheduler.on_start(repeat=3)
    @task.on_done
    def submit_task() -> None:
        x = random.randrange(10)
        task(x)

    @task.on_returned
    def print_result(result: int) -> None:
        print(result)

    results: list[int] = []

    @task.on_returned
    def save_result(result: int) -> None:
        results.append(result)

    @task.on_exception
    def print_exception(exception: BaseException) -> None:
        print(exception)

    @task.on_exception
    def stop_on_exception(exception: BaseException) -> None:
        scheduler.stop()

    @task.on_done(when=lambda: len(results) >= 50)  # (1)!
    def stop_on_50_results(result: int) -> None:
        scheduler.stop()

    scheduler.run(timeout=60)
    ```

    1. We can use the `when` keyword argument to specify a condition
       and we use an inline `#!python lambda` function to check if we have
       50 results.

!!! info "Determinism"

    It's worth noting here that for any given event the order in which callbacks
    are called is guaranteed to be the same as the order in which they were
    registered. However the order of events is not guaranteed and is subject
    to undetermined behavior based on your computations and executor.

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

    !!! warning "Memory Limits with Pynisher"

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

    !!! warning "CPU Time Limits with Pynisher"

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
