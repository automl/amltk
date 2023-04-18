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

Some popular distributions frameworks which support the `Executor` interface
or we provide an integration for.

If there's an executor background you wish to integrate, we would be happy
to consider it and greatly appreciate a PR!

####  :material-language-python: **Python**
---

Python supports the `Executor` interface natively with the
[`concurrent.futures`][concurrent.futures] module for processes with the
[`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] and
[`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] for threads.

??? example

    === "Process Pool Executor"

        ```python
        from byop.scheduling import Scheduler

        scheduler = Scheduler.with_processes(2)  # (1)!
        ```

        1. Explicitly use the `with_processes` method to create a `Scheduler` with
           a `ProcessPoolExecutor` with 2 workers.
           ```python
            from concurrent.futures import ProcessPoolExecutor
            from byop.scheduling import Scheduler

            executor = ProcessPoolExecutor(max_workers=2)
            scheduler = Scheduler(executor=executor)
           ```

    === "Thread Pool Executor"

        ```python
        from byop.scheduling import Scheduler

        scheduler = Scheduler.with_threads(2)  # (1)!
        ```

        1. Explicitly use the `with_threads` method to create a `Scheduler` with
           a `ThreadPoolExecutor` with 2 workers.
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

#### :simple-dask: [`dask`](https://distributed.dask.org/en/stable/)
---

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

#### :simple-dask: [`dask-jobqueue`](https://jobqueue.dask.org/en/latest/)
---

A package for scheduling jobs across common clusters setups such as
PBS, Slurm, MOAB, SGE, LSF, and HTCondor.


??? example

    Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
    In particular, we only control the parameter `#!python n_workers=` to
    use the [`adapt()`](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
    method, every other keyword is forwarded to the relative
    [cluster implementation](https://jobqueue.dask.org/en/latest/api.html).

    In general, you should specify the requirements of each individual worker and
    and tune your load with the `#!python n_workers=` parameter.

    If you have any tips, tricks, working setups, gotchas, please feel free
    to leave a PR or simply an issue!

    === "Slurm"

        ```python hl_lines="3 4 5 6 7 8 9"
        from byop.scheduling import Scheduler

        scheduler = Scheduler.with_slurm(
            n_workers=10,  # (1)!
            queue=...,
            cores=4,
            memory="6 GB",
            walltime="00:10:00"
        )
        ```

        1. The `n_workers` parameter is used to set the number of workers
           to start with.
           The [`adapt()`](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
           method will be called on the cluster to dynamically scale the number of workers based on
           the load.
           The `with_slurm` method will create a [`SLURMCluster`][dask_jobqueue.SLURMCluster]
           and pass it to the `Scheduler` constructor.
           ```python hl_lines="10"
           from dask_jobqueue import SLURMCluster
           from byop.scheduling import Scheduler

           cluster = SLURMCluster(
               queue=...,
               cores=4,
               memory="6 GB",
               walltime="00:10:00"
           )
           cluster.adapt(max_workers=10) (1)!
           executor = cluster.get_client().get_executor()
           scheduler = Scheduler(executor=executor)
           ```

           1. Note that we use the `n_workers` to simply execute
           this command.

        !!! warning "Running outside the login node"

            If you're running the scheduler itself in a job, this will
            not work. The scheduler itself is lightweight and can run on the
            login node without issue. However you should make sure to offload
            heavy computations to a worker.

            If you get it to work, for example in an interactive job, please
            let us know!

        !!! info "Modifying the launch command"

            On some cluster commands, you'll need to modify the launch command.
            You can use the following to do so:

            ```python
            from byop.scheduling import Scheduler

            scheduler = Scheduler.with_slurm(n_workers=..., submit_command="sbatch --extra"
            ```

    === "Others"

        Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
        and the following methods:

        * [`Scheduler.with_pbs()`][byop.scheduling.Scheduler.with_pbs]
        * [`Scheduler.with_lsf()`][byop.scheduling.Scheduler.with_lsf]
        * [`Scheduler.with_moab()`][byop.scheduling.Scheduler.with_moab]
        * [`Scheduler.with_sge()`][byop.scheduling.Scheduler.with_sge]
        * [`Scheduler.with_htcondor()`][byop.scheduling.Scheduler.with_htcondor]

#### :simple-ray: [`ray`](https://docs.ray.io/en/master/)
---

Ray is an open-source unified compute framework that makes it easy
to scale AI and Python workloads
— from reinforcement learning to deep learning to tuning,
and model serving.

??? info "In progress"

    Ray is currently in the works of supporting the Python
    `Executor` interface. See this [PR](https://github.com/ray-project/ray/pull/30826)
    for more info.

#### :simple-apacheairflow: [`airflow`](https://airflow.apache.org/)
---

Airflow is a platform created by the community to programmatically author,
schedule and monitor workflows. Their list of integrations to platforms is endless
but features compute platforms such as Kubernetes, AWS, Microsoft Azure and
GCP.

??? info "Planned"

    We plan to support `airflow` in the future. If you'd like to help
    out, please reach out to us!

---

#### :material-debug-step-over: Debugging

Sometimes you'll need to debug what's going on and remove the noise
of processes and parallelism. For this, we have implemented a very basic
[`SequentialExecutor`][byop.scheduling.SequentialExecutor] to run everything
in a sequential manner!

=== "Easy"

    ```python
    from byop.scheduling import Scheduler

    scheduler = Scheduler.with_sequential()
    ```

=== "Explicit"

    ```python
    from byop.scheduling import Scheduler, SequetialExecutor

    scheduler = Scheduler(executor=SequentialExecutor())
    ```

??? warning "Limitations"

    Due to the fact this all runs in the same process, limitations
    to [`Tasks`][byop.scheduling.Task] are likely to cause issues.
    Notably, `memory_limit`, `cpu_time_limit` and `wall_time_limit`.
    It's also likely to cause interferences with
    [`CommTask`][byop.scheduling.CommTask].



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

### Task Events
As the [`Task`][byop.scheduling.Task] is the main unit of abstraction for submitting
a function to be computed, we try to provide an extensive set of events to capture
the different events that can happen during the execution of a task.

[`Events`][byop.events.Event] are typed with the arguments that are passed to the
callback, such that they can be type check.

This looks like `#!python Event[Arg1, Arg2, ...]` such that a callback registering to such an
event should have the signature `#!python (Arg1, Arg2, ...) -> Any`, taking in the right type
of arguments and can return anything.

!!! info

    Any return value from a callback will be ignored.

Below are all the events that can be subscribed to with the [`Task`][byop.scheduling.Task]:

=== "`SUBMITTED`"

    `#!python SUBMITTED: Event[Future] = Event("task-submitted")`

    The task has been submitted to the scheduler. You can subscribe to this event
    with `on_submitted()`.

=== "`DONE`"

    `#!python DONE: Event[Future] = Event("task-done")`

    The task has finished running. You can subscribe to this event
    with `on_done()`.

=== "`CANCELLED`"

    `#!python CANCELLED: Event[Future] = Event("task-cancelled")`

    The task has been cancelled. You can subscribe to this event
    with `on_cancelled()`.

=== "`RETURNED`"

    `#!python RETURNED: Event[Any] = Event("task-returned")`

    The task has successfully returned a value. You can subscribe to this event
    with `on_returned()`.

=== "`EXCEPTION`"

    `#!python EXCEPTION: Event[BaseException] = Event("task-exception")`

    The task raised an Exception and failed to return a value.
    You can subscribe to this event with `on_exception()`.

=== "`F_RETURNED`"

    `#!python F_RETURNED: Event[Future, Any] = Event("task-future-returned")`

    The task has successfully returned a value. This also passes the future along with
    the result which can be useful during debugging and retrieving of the arguments the
    task was submitted with as a `Future` can be used as a dictionary key.
    You can subscribe to this event with `on_f_returned()`.

=== "`F_EXCEPTION`"

    `#!python F_EXCEPTION: Event[Future, BaseException] = Event("task-future-exception")`

    The task raised an Exception and failed to return a value.
    This also passes the future along with the exception which can be useful during
    debugging and retrieving of the arguments the task was submitted with as a `Future`
    can be used as a dictionary key.
    You can subscribe to this event with `on_f_exception()`.

=== "`TIMEOUT`"

    `#!python TIMEOUT: Event[Future, BaseException] = Event("task-timeout")`

    The task was given a `wall_time_limit` or `cpu_time_limit` and it timed out.
    You can subscribe to this event with `on_timeout()`.

=== "`CALL_LIMIT_REACHED`"

    `#!python CALL_LIMIT_REACHED: Event[P] = Event("task-concurrent-limit")`

    The task was submitted but reached it's maximum call limit. The callback
    will be passed the `#!python *args, **kwargs` that the task was called with.
    You can subscribe to this event with `on_concurrent_limit_reached()`.


=== "`CONCURRENT_LIMIT_REACHED`"

    `#!python CONCURRENT_LIMIT_REACHED: Event[P] = Event("task-concurrent-limit")`

    The task was submitted but reached it's maximum concurrency limit. The callback
    will be passed the `#!python *args, **kwargs` that the task was called with.
    You can subscribe to this event with `on_concurrent_limit_reached()`.

=== "`MEMORY_LIMIT_REACHED`"

    `#!python MEMORY_LIMIT_REACHED: Event[Future, BaseException] = Event("task-memory-limit")`

    The task was given a `memory_limit` and it was exceeded.
    You can subscribe to this event with `on_memory_limit_reached()`.

=== "`CPU_TIME_LIMIT_REACHED`"

    `#!python CPU_TIME_LIMIT_REACHED: Event[Future, BaseException] = Event("task-cputime-limit")`

    The task was submitted with a cpu time limit but exceeded the limit.
    You can subscribe to this event with `on_cpu_time_limit_reached()`.

=== "`WALL_TIME_LIMIT_REACHED`"

    `#!python WALL_TIME_LIMIT_REACHED: Event[Future, BaseException] = Event("task-walltime-limit")`

    The task was submitted with a wall time limit but exceeded the limit.
    You can subscribe to this event with `on_cpu_time_limit_reached()`.

## Comm Tasks
The [`CommTask`][byop.scheduling.CommTask] is a special type of task that is used
for two way communication between a worker and the server. It builds upon
a [`Task`][byop.scheduling.Task] to both specialize the signature of functions
it accepts but also to provide custom events for this kind of task.

??? warning "Local Processes Only"

    We currently use [`multiprocessing.Pipe`][multiprocessing.Pipe] to communicate
    between the worker and the scheduler. This means we are limited to local processes
    only.

    If there is interest, we could extend this to be interfaced and provide web socket
    communication as well. Please open an issue if you are interested in this or if you
    would like to contribute.

### Usage of Comm Tasks
A [`CommTask`][byop.scheduling.CommTask] relies heavily on a [`Comm`][byop.scheduling.Comm] object to
facilitate the communication between the worker and the scheduler. By using this `Comm`,
we can [`send()`][byop.scheduling.Comm.send] and [`request()`][byop.scheduling.Comm.request]
messages from the workers point of view.
These messages are then received by the scheduler and emitted as the
[`MESSAGE`][byop.scheduling.CommTask.MESSAGE] and [`REQUEST`][byop.scheduling.CommTask.REQUEST]
events respectively which both pass a [`CommTask.Msg`][byop.scheduling.CommTask.Msg] object
to the callback. This object contains the `data` that was transmitted.

Below we show an example of both `send()` and
`request()` in action.

=== "`send()`"

    ```python hl_lines="7 16 17 18 19"
    from byop.scheduling import Scheduler, CommTask, Comm

    # The function must accept a `Comm` object as the first argument
    def echoer(comm: Comm, xs: list[int]):
        with comm:  # (1)!
          for x in xs:
              comm.send(x)  # (2)!

    scheduler = Scheduler(...)
    task = CommTask(echoer, scheduler)

    @scheduler.on_start
    def start():
        task.submit([1, 2, 3, 4, 5])

    @task.on_message
    def on_message(msg: CommTask.Msg):  # (3)!
        print(f"Recieved a message {msg=}")
        print(msg.data)

    scheduler.run()
    ```

    1. The `Comm` object should be used as a context manager. This is to ensure
       that the `Comm` object is closed correctly when the function exits.
    2. Here we use the [`send()`][byop.scheduling.Comm.send] method to send a message
       to the scheduler.
    3. We can also do `#!python CommTask.Msg[int]` to specify the type of data
       we expect to receive.

=== "`request()`"

    ```python hl_lines="7 16 17 18 19"
    from byop.scheduling import Scheduler, CommTask, Comm

    # The function must accept a `Comm` object as the first argument
    def echoer(comm: Comm, n_requests: int):
        with comm:
          for _ in range(n):
              response = comm.request(n)  # (1)!

    scheduler = Scheduler(...)
    task = CommTask(echoer, scheduler)

    @scheduler.on_start
    def start():
        task.submit([1, 2, 3, 4, 5])

    @task.on_request
    def handle_request(msg: CommTask.Msg):
        print(f"Recieved request {msg=}")
        msg.respond(msg.data * 2)  # (2)!

    scheduler.run()
    ```

    1. Here we use the [`request()`][byop.scheduling.Comm.request] method to send a request
       to the scheduler with some data.
    2. We can use the [`respond()`][byop.scheduling.CommTask.Msg.respond] method to
       respond to the request with some data.

!!! tip "Identifying Workers"

    The [`CommTask.Msg`][byop.scheduling.CommTask.Msg] object also has the `future`
    attribute, which is the [`Future`][concurrent.futures.Future] object that represents
    the worker. This is hashable and usable in a dictionary as a key.

## Extending `Task`
The goal of task was to provide a simple interface for submitting functions to the `Scheduler`
but also to provide a way to extend the functionality of the `Scheduler` and `Task` objects
in a simple manner.

For an example, we will create a simple task that should accept a random seed and count
how far the seed gets without producing a number greater than `#!python 0.9`.

First we should define what a sample of this task should look like.

```python
import random

def random_task(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    count = 0
    while rng.uniform(0, 1) < 0.9:
      count += 1

    return count, seed
```

We can think about what events we would like to emit for this task. We could of course emit
an event for the task completing but we can already use [`RETURNED`][byop.scheduling.Task.RETURNED]
for this.

After some thinking, you might decide an interesting event is that we have found a seed
that produces the longest seen running count. Let's call this event, `LONGEST_RUN` which
will return the `(count, seed)`.

Let's define a simple `#!python class SeedTask(Task)` definition with this event, along with
some variable `best_count_seed` to track both the `count` and the `seed` that produced it.

=== "No types"

    ```python hl_lines="3 5 9"
    from byop.scheduling import Task, Event

    class SeedTask(Task):  # (1)!

      LONGEST_RUN = Event("longest-run") # (2)!

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed = None
    ```

    1. We inherit from the [`Task`][byop.scheduling.Task] class.
    2. We make the `LONGEST_RUN` event an attribute of the `SeedTask` class for easy access.

=== "With types"

    ```python hl_lines="3 5 9"
    from byop.scheduling import Task, Event

    class SeedTask(Task[[int], tuple[int, int]]):  # (1)!

      LONGEST_RUN: Event[tuple[int, int]] = Event("longest-run")  # (2)!

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed: tuple[int, int] | None = None
    ```

    1. The `#!python Task` class is generic and accepts two type arguments. The first
       is the type of the arguments that the task accepts and the second is the type
       of the return value, similar to a [`Callable`][typing.Callable].
       We can use this to specify the type of the arguments and the return value of the task.
    2. We make the `LONGEST_RUN` event an attribute of the `SeedTask` class for easy access.
       We can also specify the type of the event. This is useful to identify why callbacks
       which subscribe to this event should have as their signature.

Our next step is that we need to hook into the [`RETURNED`][byop.scheduling.Task.RETURNED]
event so that the `SeedTask` itself can update the `best_count_seed` variable.
If we do update the variable, we should then also `emit` the `LONGEST_RUN` event
as well as the updated variables.

=== "No types"

    ```python hl_lines="11 13 14 15 16 17"
    from byop.scheduling import Task, Event

    class SeedTask(Task):

      LONGEST_RUN = Event("longest-run")

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed = None

          self.on_returned(self.check_longest_run)  # (1)!

      def check_longest_run(self, result):
          count, seed = result
          if self.best_count_seed is None or count > self.best_count_seed[0]:
              self.best_count_seed = (count, seed)
              self.emit(self.LONGEST_RUN, (count, seed))  # (2)!
    ```

    1. We can use the functional form of subscribing to `on_returned()` to
       register a callback function that will be called when the task returns.
       The callback function will be passed the return value of the task.
    2. We can use `emit()` to emit the `LONGEST_RUN` event.
       The callback functions will be passed the `(count, seed)`.

=== "With types"

    ```python hl_lines="11 13 14 15 16 17"
    from byop.scheduling import Task, Event

    class SeedTask(Task[[int], tuple[int, int]]):

      LONGEST_RUN: Event[tuple[int, int]] = Event("longest-run")

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed = None

          self.on_returned(self.check_longest_run)  # (1)!

      def check_longest_run(self, result: tuple[int, int]) -> None:
          count, seed = result
          if self.best_count_seed is None or count > self.best_count_seed[0]:
              self.best_count_seed = (count, seed)
              self.emit(self.LONGEST_RUN, (count, seed))  # (2)!
    ```

    1. We can use the functional form of subscribing to `on_returned()` to
       register a callback function that will be called when the task returns.
       The callback function will be passed the return value of the task.
    2. We can use `emit()` to emit the `LONGEST_RUN` event.
       The callback functions will be passed the `(count, seed)`.


Now let's use this task! We'll start simply by just running it once on `scheduler.on_start`
and printing the results.

```python hl_lines="17 18 19 20 21 23 24 25 26 27"
import random
from byop.scheduling import Scheduler, Task, Event

def random_task(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    count = 0
    while rng.uniform(0, 1) < 0.9:
      count += 1

    return count, seed

scheduler = Scheduler.with_processes(4)
task = SeedTask(random_task, scheduler)

counts: dict[int, int] = {}

@scheduler.on_start
def on_start() -> None:
    seed = len(counts)
    counts[seed] = None
    task(seed)

@task.on_returned
def on_returned(result):
    count, seed = result
    counts[seed] = count
    print(count, seed)

scheduler.run()

print(task.best_count_seed)
```

Now with some careful though readers may have noticed we now have to implement something
along the lines of `on_longest_run()` to be able to subscribe callbacks to the emitted event.
This is rather a repetitive process when defining new tasks so we provide a convenience function
for this, namely [`subscriber()`][byop.scheduling.Task.subscriber]. This allows us to conveniently
make a [`Subscriber`][byop.events.Subscriber] which handles all of this for us.

=== "No types"

    ```python hl_lines="13 19"
    from byop.scheduling import Scheduler, Task, Event

    class SeedTask(Task):

      LONGEST_RUN = Event("longest-run")

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed = None

          self.on_returned(self.check_longest_run)

          self.on_longest_run = self.subscriber(self.LONGEST_RUN)

      def check_longest_run(self, result):
          count, seed = result
          if self.best_count_seed is None or count > self.best_count_seed[0]:
              self.best_count_seed = (count, seed)
              self.on_longest_run.emit((count, seed))
    ```

    1. We can also use the [`Subscriber.emit()`][byop.events.Subscriber.emit] method
       to automatically emit the corresponding event but this is only for convenience
       and is up to you.

=== "With types"

    ```python hl_lines="13 19"
    from byop.scheduling import Scheduler, Task, Event, Subscriber

    class SeedTask(Task[[int], tuple[int, int]]):

      LONGEST_RUN: Event[tuple[int, int]] = Event("longest-run")

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_count_seed = None

          self.on_returned(self.check_longest_run)

          self.on_longest_run = self.subscriber(self.LONGEST_RUN)

      def check_longest_run(self, result: tuple[int, int]) -> None:
          count, seed = result
          if self.best_count_seed is None or count > self.best_count_seed[0]:
              self.best_count_seed = (count, seed)
              self.on_longest_run.emit((count, seed))  # (1)!
    ```


    1. We can also use the [`Subscriber.emit()`][byop.events.Subscriber.emit] method
       to automatically emit the corresponding event but this is only for convenience
       and is up to you.

Now we can use this task as before but now we can subscribe to the `on_longest_run` event.

```python hl_lines="29 30 31 32"
import random
from byop.scheduling import Scheduler, Task, Event

def random_task(seed):
    rng = random.Random(seed)
    count = 0
    while rng.uniform(0, 1) < 0.9:
      count += 1

    return count, seed

scheduler = Scheduler.with_processes(4)
task = SeedTask(random_task, scheduler)

counts = {}

@scheduler.on_start
def on_start():
    seed = len(counts)
    counts[seed] = None
    task(seed)

@task.on_returned
def on_returned(result):
    count, seed = result
    counts[seed] = count
    print(count, seed)

@task.on_longest_run
def on_longest_run(result):
    count, seed = result
    print("Longest run so far:", count, seed)

scheduler.run()
```

This short demonstration showed you how to define your own `Task` and custom `Event`s
and seamlessly hook them into the scheduler and event system. You may wish to check
out [`Trial.Task`][byop.optimization.Trial.Task] for a more complete example of how we
extended `Task` to implement the optimization part of AutoML-Toolkit. You may also wish to
build off of [`CommTask`][byop.scheduling.CommTask] for defining tasks which communicate but
that exercise is left to the reader.

---

This concludes the guide to how AutoML-Toolkit works. If you have any questions or
comments please feel free to open an issue!
