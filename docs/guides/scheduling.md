AutoML-toolkit was designed to make offloading computation
away from the main process __easy__, to foster increased ability for
interact-ability deployment and control. At the same time,
we wanted to have an event based system to manage the complexity
that comes with AutoML systems, all while making the API intuitive
and extensible.

By the end of this guide, we hope that the following code, its options
and its inner working become easy to understand.

```python exec="true" source="tabbed-left" html="True" title="Scheduler"
from amltk import Scheduler

# Some function to offload to compute
def collatz(n: int) -> int:
    is_even = (n % 2 == 0)
    return int(n / 2) if is_even else int(3 * n + 1)
from amltk._doc import make_picklable; make_picklable(collatz)  # markdown-exec: hide

# Setup the scheduler and create a "task"
scheduler = Scheduler.with_processes(1)
task = scheduler.task(collatz)

answers = []

# Tell the scheduler what to do when it starts
@scheduler.on_start
def start_computing() -> None:
    answers.append(12)
    task(12)  # Launch the task with the argument 12

# Tell the scheduler what to do when the task returns
@task.on_result
def compute_next(_, next_n: int) -> None:
    answers.append(next_n)
    if next_n != 1:
        task(next_n)

# Run the scheduler
scheduler.run(timeout=1)  # One second timeout
print(answers)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```


We start by introducing the engine, the [`Scheduler`][amltk.scheduling.Scheduler]
and how this interacts with python's built-in [`Executor`][concurrent.futures.Executor]
interface to offload compute to processes, cluster nodes, or even cloud resources.

However, the `Scheduler` is rather useless without some fuel. For this,
we present [`Tasks`][amltk.scheduling.Task], the computational task to
perform with the `Scheduler` and start the system's gears turning.

Finally, we show the [`Event`][amltk.events.Event] and how you can use this with
an [`Emitter`][amltk.events.Emitter] to create your own event-driven systems.

??? tip "`rich` printing"

    To get the same output locally (terminal or Notebook), you can either
    call `thing.__rich()__`, use `from rich import print; print(thing)`
    or in a Notebook, simply leave it as the last object of a cell.

## Scheduler
The core engine of the AutoML-Toolkit is the [`Scheduler`][amltk.scheduling.Scheduler].
It purpose it to allow you to create workflows in an event driven manner. It does
this by allowing you to [`submit()`][amltk.scheduling.Scheduler.submit] functions
with arguments to be computed in the background, while the main process can continue
to do other work. Once this computation has completed, you can react with various
callbacks, most likely to submit more computations.

??? tip "Sounds like `asyncio`?"

    If you're familiar with pythons `await/async` syntax, then this description
    might sound similar. The `Scheduler` is powered by an asynchronous event loop
    but hides this complexity in it's API. We do have an asynchronous API which
    we will discuss later.

### Backend

The first thing to do is define where this computation should happen.
A [`Scheduler`][amltk.scheduling.Scheduler] builds upon, an
[`Executor`][concurrent.futures.Executor],
an interface provided by python's [`concurrent.futures`][concurrent.futures]
module. This interface is used to abstract away the details of how the
computation is actually performed. This allows us to easily switch between
different backends, such as threads, processes, clusters, cloud resources,
or even custom backends.

The simplest one is a [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor]
which will create a pool of processes to run the compute in parallel.

```python exec="true" source="material-block" html="True"
from concurrent.futures import ProcessPoolExecutor
from amltk.scheduling import Scheduler

scheduler = Scheduler(
    executor=ProcessPoolExecutor(max_workers=2),
)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

We provide convenience functions for some common backends, such as
[`with_processes(max_workers=2)`][amltk.scheduling.Scheduler.with_processes]
which does exactly this.

!!! tip "Builtin backends"

    If there's any executor background you wish to integrate, we would
    be happy to consider it and greatly appreciate a PR!

    === ":material-language-python: `Python`"

        Python supports the `Executor` interface natively with the
        [`concurrent.futures`][concurrent.futures] module for processes with the
        [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] and
        [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] for threads.

        !!! example

            === "Process Pool Executor"

                ```python
                from amltk.scheduling import Scheduler

                scheduler = Scheduler.with_processes(2)  # (1)!
                ```

                1. Explicitly use the `with_processes` method to create a `Scheduler` with
                   a `ProcessPoolExecutor` with 2 workers.
                   ```python
                    from concurrent.futures import ProcessPoolExecutor
                    from amltk.scheduling import Scheduler

                    executor = ProcessPoolExecutor(max_workers=2)
                    scheduler = Scheduler(executor=executor)
                   ```

            === "Thread Pool Executor"

                ```python
                from amltk.scheduling import Scheduler

                scheduler = Scheduler.with_threads(2)  # (1)!
                ```

                1. Explicitly use the `with_threads` method to create a `Scheduler` with
                   a `ThreadPoolExecutor` with 2 workers.
                   ```python
                    from concurrent.futures import ThreadPoolExecutor
                    from amltk.scheduling import Scheduler

                    executor = ThreadPoolExecutor(max_workers=2)
                    scheduler = Scheduler(executor=executor)
                   ```

                !!! danger "Why to not use threads"

                    Python also defines a [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor]
                    but there are some known drawbacks to offloading heavy compute to threads. Notably,
                    there's no way in python to terminate a thread from the outside while it's running.

    === ":simple-dask: `dask`"

        [Dask](https://distributed.dask.org/en/stable/) and the supporting extension [`dask.distributed`](https://distributed.dask.org/en/stable/)
        provide a robust and flexible framework for scheduling compute across workers.

        !!! example

            ```python hl_lines="5"
            from dask.distributed import Client
            from amltk.scheduling import Scheduler

            client = Client(...)
            executor = client.get_executor()
            scheduler = Scheduler(executor=executor)
            ```

    === ":simple-dask: `dask-jobqueue`"

        [`dask-jobqueue`](https://jobqueue.dask.org/en/latest/) is a package
        for scheduling jobs across common clusters setups such as
        PBS, Slurm, MOAB, SGE, LSF, and HTCondor.


        !!! example

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
                from amltk.scheduling import Scheduler

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
                   method will be called on the cluster to dynamically scale up to `#!python n_workers=` based on
                   the load.
                   The `with_slurm` method will create a [`SLURMCluster`][dask_jobqueue.SLURMCluster]
                   and pass it to the `Scheduler` constructor.
                   ```python hl_lines="10"
                   from dask_jobqueue import SLURMCluster
                   from amltk.scheduling import Scheduler

                   cluster = SLURMCluster(
                       queue=...,
                       cores=4,
                       memory="6 GB",
                       walltime="00:10:00"
                   )
                   cluster.adapt(max_workers=10)
                   executor = cluster.get_client().get_executor()
                   scheduler = Scheduler(executor=executor)
                   ```

                !!! warning "Running outside the login node"

                    If you're running the scheduler itself in a job, this may not
                    work on some cluster setups. The scheduler itself is lightweight
                    and can run on the login node without issue.
                    However you should make sure to offload heavy computations
                    to a worker.

                    If you get it to work, for example in an interactive job, please
                    let us know!

                !!! info "Modifying the launch command"

                    On some cluster commands, you'll need to modify the launch command.
                    You can use the following to do so:

                    ```python
                    from amltk.scheduling import Scheduler

                    scheduler = Scheduler.with_slurm(n_workers=..., submit_command="sbatch --extra"
                    ```

            === "Others"

                Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
                and the following methods:

                * [`Scheduler.with_pbs()`][amltk.scheduling.Scheduler.with_pbs]
                * [`Scheduler.with_lsf()`][amltk.scheduling.Scheduler.with_lsf]
                * [`Scheduler.with_moab()`][amltk.scheduling.Scheduler.with_moab]
                * [`Scheduler.with_sge()`][amltk.scheduling.Scheduler.with_sge]
                * [`Scheduler.with_htcondor()`][amltk.scheduling.Scheduler.with_htcondor]

    === ":octicons-gear-24: `loky`"

        [Loky](https://loky.readthedocs.io/en/stable/API.html) is the default backend executor behind
        [`joblib`](https://joblib.readthedocs.io/en/stable/), the parallelism that
        powers scikit-learn.

        !!! example "Scheduler with Loky Backend"

            === "Simple"

                ```python
                from amltk import Scheduler

                # Pass any arguments you would pass to `loky.get_reusable_executor`
                scheduler = Scheduler.with_loky(...)
                ```


            === "Explicit"

                ```python
                import loky
                from amltk import Scheduler

                scheduler = Scheduler(executor=loky.get_reusable_executor(...))
                ```

        !!! warning "BLAS numeric backend"

            The loky executor seems to pick up on a different BLAS library (from scipy)
            which is different than those used by jobs from something like a `ProcessPoolExecutor`.

            This is likely not to matter for a majority of use-cases.

    === ":simple-ray: `ray`"

        [Ray](https://docs.ray.io/en/master/) is an open-source unified compute framework that makes it easy
        to scale AI and Python workloads
        — from reinforcement learning to deep learning to tuning,
        and model serving.

        !!! info "In progress"

            Ray is currently in the works of supporting the Python
            `Executor` interface. See this [PR](https://github.com/ray-project/ray/pull/30826)
            for more info.

    === ":simple-apacheairflow: `airflow`"

        [Airflow](https://airflow.apache.org/) is a platform created by the community to programmatically author,
        schedule and monitor workflows. Their list of integrations to platforms is endless
        but features compute platforms such as Kubernetes, AWS, Microsoft Azure and
        GCP.

        !!! info "Planned"

            We plan to support `airflow` in the future. If you'd like to help
            out, please reach out to us!

    === ":material-debug-step-over: Debugging"

        Sometimes you'll need to debug what's going on and remove the noise
        of processes and parallelism. For this, we have implemented a very basic
        [`SequentialExecutor`][amltk.scheduling.SequentialExecutor] to run everything
        in a sequential manner!

        === "Easy"

            ```python
            from amltk.scheduling import Scheduler

            scheduler = Scheduler.with_sequential()
            ```

        === "Explicit"

            ```python
            from amltk.scheduling import Scheduler, SequetialExecutor

            scheduler = Scheduler(executor=SequentialExecutor())
            ```

        !!! warning "Recursion"

            If you use The `SequentialExecutor`, be careful that the stack
            of function calls can get quite large, quite quick. If you are
            using this for debugging, keep the number of submitted tasks
            from callbacks small and focus in on debugging. If using this
            for sequential ordering of operations, prefer to use
            `with_processes(1)` as this will still maintain order but not
            have these stack issues.


### Running the Scheduler

You may have noticed from the above example that there are many events the shceduler will emit,
such as `@on_start` or `@on_future_done`. One particularly important one is
[`@on_start`][amltk.scheduling.Scheduler.on_start], an event to signal
the scheduler has started and ready to accept tasks.

```python exec="true" source="material-block" html="True"
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(1)

@scheduler.on_start
def print_hello() -> None:
    print("hello")
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

From the output, we can see that the `print_hello()` function was registered
to the event `@on_start`, but it was never called and no `#!python "hello"` printed.

For this to happen, we actually have to [`run()`][amltk.scheduling.Scheduler.run] the scheduler.

```python exec="true" source="material-block" html="True"
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(1)

@scheduler.on_start
def print_hello() -> None:
    print("hello")

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

Now the output will show a little yellow number next to the `@on_start`
and the `print_hello()`, indicating that event was triggered and the callback
was called.

You can subscribe multiple callbacks to the same event and they will each be called
in the order they were registered.

```python exec="true" source="material-block" html="True"
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(1)

@scheduler.on_start
def print_hello_1() -> None:
    print("hello 1")

def print_hello_2() -> None:
    print("hello 2")

scheduler.on_start(print_hello_2)  # You can also register without a decorator

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

### Submitting Compute
The `Scheduler` exposes a simple [`submit()`][amltk.scheduling.Scheduler.submit]
method which allows you to submit compute to be performed **while the scheduler is running**.

While we will later visit the [`Task`][amltk.scheduling.Task] class for
defining these units of compute, is benficial to see how the `Scheduler`
operates directly with `submit()`, without abstractions.

In the below example, we will use the
[`@on_future_result`][amltk.scheduling.Scheduler.on_future_result]
event to submit more compute once the previous computation has returned a result.

```python exec="true" source="material-block" html="True" hl_lines="10 13 18"
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(2)

def expensive_function(x: int) -> int:
    return 2 ** x
from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

@scheduler.on_start
def submit_calculations() -> None:
    scheduler.submit(expensive_function, 2)  # Submit compute

# Called when the submitted function is done
@scheduler.on_future_result
def print_result(_, result: int) -> None:
    print(result)
    if result < 10:
        scheduler.submit(expensive_function, result)

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

??? tip "What's a `Future`?"

    A [`Future`][asyncio.Future] is a special object which represents the result
    of an asynchronous computation. It's an object that can be queried for
    its result/exception of some computation which may not have completed yet.

### Scheduler Events
We won't cover all possible scheduler events but we provide the complete list here:

=== "`@on_start`"

    ::: amltk.scheduling.Scheduler.on_start

=== "`@on_future_result`"

    ::: amltk.scheduling.Scheduler.on_future_result

=== "`@on_future_exception`"

    ::: amltk.scheduling.Scheduler.on_future_exception

=== "`@on_future_submitted`"

    ::: amltk.scheduling.Scheduler.on_future_submitted

=== "`@on_future_done`"

    ::: amltk.scheduling.Scheduler.on_future_done

=== "`@on_future_cancelled`"

    ::: amltk.scheduling.Scheduler.on_future_cancelled

=== "`@on_timeout`"

    ::: amltk.scheduling.Scheduler.on_timeout

=== "`@on_stop`"

    ::: amltk.scheduling.Scheduler.on_stop

=== "`@on_finishing`"

    ::: amltk.scheduling.Scheduler.on_finishing

=== "`@on_finished`"

    ::: amltk.scheduling.Scheduler.on_finished

=== "`@on_empty`"

    ::: amltk.scheduling.Scheduler.on_empty

We can access all the counts of all events through the
[`scheduler.event_counts`][amltk.events.Emitter.event_counts] property.
This is a `dict` which has the events as keys and the amount of times
it was emitted as the values.

!!! tip "Determinism"

    It's worth noting that even though we are using an event based system, we
    are still guaranteed deterministic execution of the callbacks for any given
    event. The source of indeterminism is the order in which events are emitted,
    this is determined entirely by your compute functions themselves.


### Controlling Callbacks
There's a few parameters you can pass to any event subscriber
such as `@on_start` or `@on_future_result`.
These control the behavior of what happens when its event is fired and can
be used to control the flow of your system.

You can find their docs here [`Emitter.on()`][amltk.events.Emitter.on].

=== "`repeat=`"

    Repeat the callback a certain number of times, every time the event is emitted.

    ```python exec="true" source="material-block" html="True" hl_lines="6"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    # Print "hello" 3 times when the scheduler starts
    @scheduler.on_start(repeat=3)
    def print_hello() -> None:
        print("hello")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

=== "`limit=`"

    Limit the number of times a callback can be called, after which, the callback
    will be ignored.

    ```python exec="true" source="material-block" html="True" hl_lines="13"
    from asyncio import Future
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(2)

    def expensive_function(x: int) -> int:
        return x ** 2
    from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

    @scheduler.on_start
    def submit_calculations() -> None:
        scheduler.submit(expensive_function, 2)

    @scheduler.on_future_result(limit=3)
    def print_result(future, result) -> None:
        scheduler.submit(expensive_function, 2)

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

=== "`when=`"

    A callable which takes no arguments and returns a `bool`. The callback
    will only be called when the `when` callable returns `True`.

    Below is a rather contrived example, but it shows how we can use the
    `when` parameter to control when the callback is called.

    ```python exec="true" source="material-block" html="True" hl_lines="8 12"
    import random
    from amltk.scheduling import Scheduler

    LOCALE = random.choice(["English", "German"])

    scheduler = Scheduler.with_processes(1)

    @scheduler.on_start(when=lambda: LOCALE == "English")
    def print_hello() -> None:
        print("hello")

    @scheduler.on_start(when=lambda: LOCALE == "German")
    def print_guten_tag() -> None:
        print("guten tag")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

=== "`every=`"

    Only call the callback every `every` times the event is emitted. This
    includes the first time it's called.

    ```python exec="true" source="material-block" html="True" hl_lines="6"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    # Print "hello" only every 2 times the scheduler starts.
    @scheduler.on_start(every=2)
    def print_hello() -> None:
        print("hello")

    # Run the scheduler 5 times
    scheduler.run()
    scheduler.run()
    scheduler.run()
    scheduler.run()
    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

### Stopping the Scheduler
There are a few ways the `Scheduler` will stop. The one we have implicitly
been using this whole time is when the `Scheduler` has run out of events
to process with no compute left to perform. This is the default behavior
but can be controlled with [`run(end_on_empty=False)`][amltk.scheduling.Scheduler.run].

However there are more explicit methods.

=== "`scheduler.stop()`"

    You can explicitly call [`stop()`][amltk.scheduling.Scheduler.stop]
    from aywhere on the `Scheduler` to stop it. By default this will
    wait for any currently running compute to finish but you can inform the
    scheduler to stop immediately with [`run(wait=False)`][amltk.scheduling.Scheduler.run].

    You'll notice this in the event count of the Scheduler where the event
    `@on_future_cancelled` was fired.

    ```python exec="true" source="material-block" html="True" hl_lines="13-15"
    import time
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    def expensive_function(sleep_for: int) -> None:
        time.sleep(sleep_for)
    from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

    @scheduler.on_start
    def submit_calculations() -> None:
        scheduler.submit(expensive_function, sleep_for=10)

    @scheduler.on_future_submitted
    def stop_the_scheduler(_) -> None:
        scheduler.stop()

    scheduler.run(wait=False)
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

=== "`scheduler.run(timeout=...)`"

    You can also tell the `Scheduler` to stop after a certain amount of time
    with the `timeout=` argument to [`run()`][amltk.scheduling.Scheduler.run].

    This will also trigger the `@on_timeout` event as seen in the `Scheduler` output.

    ```python exec="true" source="material-block" html="True" hl_lines="19"
    import time
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    def expensive_function() -> None:
        time.sleep(0.1)
        return 42
    from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

    @scheduler.on_start
    def submit_calculations() -> None:
        scheduler.submit(expensive_function)

    # The will endlessly loop the scheduler
    @scheduler.on_future_done
    def submit_again(future: Future) -> None:
        scheduler.submit(expensive_function)

    scheduler.run(timeout=1)  # End after 1 second
    from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
    ```

### Exceptions
Dealing with exceptions is an important part of any AutoML system. It is important
to clarify that there's two kinds of exceptions that can occur within the Scheduler.

The 1st kind that can happen is within some function submitted with
[`submit()`][amltk.scheduling.Scheduler.submit]. When this happens,
the `@on_future_exception` will be emitted, passing the exception to the callback.

By default, the `Scheduler` will then raise the exception that occured up to your program
and end it's computations. This is done by setting
[`#!python run(on_exception="raise")`][amltk.scheduling.Scheduler],
the default, but it also takes two other possibilities:

* `#!python "ignore"` - Just ignore the exception and keep running.
* `#!python "end"` - Stop the scheduler but don't raise it.

One example is to just `stop()` the scheduler when some exception occurs.

```python exec="true" source="material-block" html="True" hl_lines="12-14"
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(1)

def failing_compute_function(err_msg: str) -> None:
    raise ValueError(err_msg)
from amltk._doc import make_picklable; make_picklable(failing_compute_function)  # markdown-exec: hide

@scheduler.on_start
def submit_calculations() -> None:
    scheduler.submit(failing_compute_function, "Failed!")

@scheduler.on_future_exception
def stop_the_scheduler(future: Future, exception: Exception) -> None:
    print("Got exception {exception}")
    scheduler.stop()  # You can optionally pass `exception=` for logging purposes.

scheduler.run(on_exception="ignore")  # Scheduler will not stop because of the error
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

The second kind of exception that can happen is one that happens in the main process.
For example this could happen in one of your callbacks or in the `Scheduler` itself (please raise an issue if this occurs!).
By default when you call [`run()`][amltk.scheduling.Scheduler.run] it will set
`#!python run(on_exception="raise")` and raise the exception that occured, with its traceback.
This is to help you debug your program.

You may also use `#!python run(on_exception="end")` which will just end the `Scheduler` and raise no exception,
or use `#!python run(on_exception="ignore")`, in which case the `Scheduler` will continue on with whatever events
are next to process.

## Tasks
Now that we have seen how the [`Scheduler`][amltk.scheduling.Scheduler] works,
we can look at the [`Task`][amltk.scheduling.Task], a wrapper around a function
that you'll want to submit to the `Scheduler`. The preffered way to create one
of these `Tasks` is to use [`scheduler.task(function)`][amltk.scheduling.Scheduler.task].

### Running a task
In the following example, we will create a task for the scheduler and attempt to
call it. This task will be run by the backend specified.

```python exec="true" source="material-block" result="python"
from amltk import Scheduler

# Some function to offload to compute
def collatz(n: int) -> int:
    is_even = (n % 2 == 0)
    return int(n / 2) if is_even else int(3 * n + 1)
from amltk._doc import make_picklable; make_picklable(collatz)  # markdown-exec: hide

scheduler = Scheduler.with_processes(1)

# Creating a "task"
collatz_task = scheduler.task(collatz)

try:
    collatz_task(5)
except Exception as e:
    print(f"{type(e)}: {e}")
```

As you can see, we **can not** submit tasks before the scheduler is running. This
is because the backend that it's running on usually has to setup and teardown when
`scheduler.run()` is called.

The proper approach would be to do the following:

```python exec="true" source="material-block" html="True"
from amltk import Scheduler

# Some function to offload to compute
def collatz(n: int) -> int:
    is_even = (n % 2 == 0)
    return int(n / 2) if is_even else int(3 * n + 1)
from amltk._doc import make_picklable; make_picklable(collatz)  # markdown-exec: hide

# Setup the scheduler and create a "task"
scheduler = Scheduler.with_processes(1)
collatz_task = scheduler.task(collatz)

@scheduler.on_start
def launch_initial_task() -> None:
    collatz_task(5)

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

### Task Specific Events
As you may have noticed, we can see the `Task` itself in the `Scheduler` as well as the
events it defines. This allows us to react to certain tasks themselves, and not generally
everything that may pass through the `Scheduler`.

In the below example, we'll do two things. First is we'll create a `Task` and react to
it's events, but also use the `Scheduler` directly and use `submit()`. Then we'll see
how the callbacks reacted to different events.


```python exec="true" source="material-block" html="True"
from amltk import Scheduler

def echo(msg: str) -> str:
    return msg
from amltk._doc import make_picklable; make_picklable(echo)  # markdown-exec: hide

scheduler = Scheduler.with_processes(1)
echo_task = scheduler.task(echo)

# Launch the task and do a raw `submit()` with the Scheduler
@scheduler.on_start
def launch_initial_task() -> None:
    echo_task("hello")
    scheduler.submit(echo, "hi")

# Callback for anything result from the scheduler
@scheduler.on_future_result
def from_scheduler(_, msg: str) -> None:
    print(f"result_from_scheduler {msg}")

# Callback for specifically results from the `echo_task`
@echo_task.on_result
def from_task(_, msg: str) -> None:
    print(f"result_from_scheduler {msg}")

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

We can see in the output of the above code that the `@scheduler.on_future_result` was called twice,
meaning our callback `#!python def from_scheduler()` was called twice,
one for the result of `#!python echo_task("hello")` and the other
from `#!python scheduler.submit(echo, "hi")`. On the other hand, the event `@task.on_result`
was only called once, meaning our callback `#!python def from_task()` was only called once.

In practice, you will likely need to define a variety of tasks for your AutoML System and
having dedicated code to respond to individual tasks is of critical importance. This
can even allow you to chain the results of one task into another, and define more complex
workflows.

The below example shows how you can define two tasks with the scheduler and have
certain callbacks for different tasks, or even share callbacks between them!

```python exec="true" source="material-block" html="True"
from amltk import Scheduler

def expensive_thing_1(x: int) -> int:
    return x * 2
from amltk._doc import make_picklable; make_picklable(expensive_thing_1)  # markdown-exec: hide

def expensive_thing_2(x: int) -> int:
    return x ** 2
from amltk._doc import make_picklable; make_picklable(expensive_thing_2)  # markdown-exec: hide

# Create a scheduler and 2 tasks
scheduler = Scheduler.with_processes(1)
task_1 = scheduler.task(expensive_thing_1)
task_2 = scheduler.task(expensive_thing_1)

# A list of things we want to compute
items = iter([1, 2, 3])

@scheduler.on_start
def submit_initial() -> None:
    next_item = next(items)
    task_1(next_item)

@task_1.on_result
def submit_task_2_with_results_of_task_1(_, result: int) -> None:
    """When task_1 returns, send the result to task_2"""
    task_2(result)

@task_1.on_result
def submit_task_1_with_next_item(_, result: int) -> None:
    """When task_1 returns, launch it again with the next items"""
    next_item = next(items, None)
    if next_item is not None:
        task_1(next_item)
        return

    print("Done!")

# You may share callbacks for the two tasks
@task_1.on_exception
@task_2.on_exception
def handle_task_exception(_, exception: BaseException) -> None:
    print(f"A task errored! {exception}")

scheduler.run()
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

## Task Plugins
Another benefit of [`Task`][amltk.scheduling.Task] objects is that we can attach
a [`TaskPlugin`][amltk.scheduling.TaskPlugin] to them. These plugins can automate control
behaviour of tasks, either through preventing their execution,
modifying the function and its arugments or even attaching plugin specific events!

For a complete reference, please see the [plugin reference page](site:reference/plugins).

### Call Limiter
Perhaps one of the more useful plugins, at least when designing an AutoML System is the
[`CallLimiter`][amltk.scheduling.task_plugin.CallLimiter] plugin. This can help you control
both it's concurrency or the absolute limit of how many times a certain task can be
successfully submitted.

In the following contrived example, we will setup a `Scheduler` with 2 workers and attempt
to submit a `Task` 4 times in rapid succession. However we have the constraint that we
only ever want 2 of these tasks running at a given time. Let's see how we could achieve that.

```python exec="true" source="material-block" html="True" hl_lines="9"
from amltk import Scheduler, CallLimiter

def my_func(x: int) -> int:
    return x
from amltk._doc import make_picklable; make_picklable(my_func)  # markdown-exec: hide

scheduler = Scheduler.with_processes(2)

# Specify a concurrency limit of 2
task = scheduler.task(my_func, plugins=CallLimiter(max_concurrent=2))

# A list of 10 things we want to compute
items = iter(range(10))
results = []

@scheduler.on_start(repeat=4)  # Repeat callback 4 times
def submit() -> None:
    next_item = next(items)
    task(next_item)

@task.on_result
def record_result(_, result: int) -> None:
    results.append(result)

@task.on_result
def launch_another(_, result: int) -> None:
    next_item = next(items, None)
    if next_item is not None:
        task(next_item)

scheduler.run()
print(results)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

You can notice that this limiting worked, given the numbers `#!python 2` and `#!python 3`
were skipped and not printed. As expected, we successfully launched the task with both 
`#!python 0` and `#!python 1` but as these tasks were not done processing, the `CallLimiter`
kicks in and prevents the other two.

A natural extension to ask is then, "how do we requeue these?". Well lets take a look at the above
output. The plugin has added three new events to `Task`, namely
`@call-limit-reached`, `@concurrent-limit-reached` and `@disabled-due-to-running-task`.

To subscribe to these _extra_ events (or any for that matter), we can use
the [`task.on()`][amltk.scheduling.Task]
method. Below is the same example except here we respond to `@call-limit-reached`
and requeue the submissions that failed.

```python exec="true" source="material-block" html="True" hl_lines="11 19-21"
from amltk import Scheduler, CallLimiter
from amltk.types import Requeue

def my_func(x: int) -> int:
    return x
from amltk._doc import make_picklable; make_picklable(my_func)  # markdown-exec: hide

scheduler = Scheduler.with_processes(2)
task = scheduler.task(my_func, plugins=CallLimiter(max_concurrent=2))

# A list of 10 things we want to compute
items = Requeue(range(10))  # A convenience type that you can requeue/append to
results = []

@scheduler.on_start(repeat=4)  # Repeat callback 4 times
def submit() -> None:
    next_item = next(items)
    task(next_item)

@task.on("concurrent-limit-reached")
def add_back_to_queue(task: Task, x: int) -> None:
    items.requeue(x)  # Put x back at the start of the queue

@task.on_result
def record_result(_, result: int) -> None:
    results.append(result)

@task.on_result
def launch_another(_, result: int) -> None:
    next_item = next(items, None)
    if next_item is not None:
        task(next_item)

scheduler.run()
print(results)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

### Creating Your Own Task Plugins
The [`CallLimiter`][amltk.scheduling.task_plugin.CallLimiter] plugin is a good example
of what you can achieve with a [`TaskPlugin`][amltk.scheduling.task_plugin.TaskPlugin]
and serves as a reference point for how you can add new events and control
task submission.

Another good example is the [`PynisherPlugin`][amltk.pynisher.pynisher_task_plugin.PynisherPlugin] which
wraps a `Task` when it's submitted, allowing you to limit memory and wall clock time
of your compute functions, in a cross-platform manner.

If you have any cool new plugins, we'd love to hear about them!
Please see the [plugin reference page](site:reference/plugins.md) for more.

## Emitters and the Event System

!!! todo "TODO"

    This section should contain a little overview of how the
    [`Emitter`][amltk.events.Emitter] class works as it's the main
    layer through which objects register and emit events, often
    by creating a [`subscriber()`][amltk.events.Emitter.subscriber].

    This should also briefly mention what an [`Event`][amltk.events.Event]
    is.

### Events
TODO

### Emitters
TODO
