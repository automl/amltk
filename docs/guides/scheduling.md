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
    task.submit(12)  # Launch the task with the argument 12

# Tell the scheduler what to do when the task returns
@task.on_result
def compute_next(_, next_n: int) -> None:
    answers.append(next_n)
    if scheduler.running() or next_n != 1:
        task.submit(next_n)

# Run the scheduler
scheduler.run(timeout=1)  # One second timeout
print(answers)
from amltk._doc import doc_print; doc_print(print, scheduler, fontsize="small")  # markdown-exec: hide
```


We start by introducing the engine, the [`Scheduler`][amltk.scheduling.Scheduler]
and how this interacts with python's built-in [`Executor`][concurrent.futures.Executor]
interface to offload compute to processes, cluster nodes, or even cloud resources.

However, the `Scheduler` is rather useless without some fuel. For this,
we present [`Tasks`][amltk.scheduling.Task], the computational task to
perform with the `Scheduler` and start the system's gears turning.

??? tip "`rich` printing"

    To get the same output locally (terminal or Notebook), you can either
    call `thing.__rich()__`, use `from rich import print; print(thing)`
    or in a Notebook, simply leave it as the last object of a cell.

    You'll have to install with `amltk[jupyter]` or
    `pip install rich[jupyter]` manually.k

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

!!! info inline end "Available Executors"

    You can find a list of these in our
    [executor reference](site:reference/scheduling/executors.md).

The simplest one is a [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor]
which will create a pool of processes to run the compute in parallel. We provide
a convenience function for this as
[`Scheduler.with_processes()`][amltk.scheduling.Scheduler.with_processes]
well as some other builder

```python exec="true" source="material-block" html="True"
from concurrent.futures import ProcessPoolExecutor
from amltk.scheduling import Scheduler

scheduler = Scheduler.with_processes(2)
from amltk._doc import doc_print; doc_print(print, scheduler)  # markdown-exec: hide
```

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

!!! tip "Determinism"

    It's worth noting that even though we are using an event based system, we
    are still guaranteed deterministic execution of the callbacks for any given
    event. The source of indeterminism is the order in which events are emitted,
    this is determined entirely by your compute functions themselves.

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
Here are some of the possible `@events` a `Scheduler` can emit, but
please visit the [scheduler reference](site:reference/scheduling/scheduler.md)
for a complete list.

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
[`scheduler.event_counts`][amltk.scheduling.events.Emitter.event_counts] property.
This is a `dict` which has the events as keys and the amount of times
it was emitted as the values.

### Controlling Callbacks
There's a few parameters you can pass to any event subscriber
such as `@on_start` or `@on_future_result`.
These control the behavior of what happens when its event is fired and can
be used to control the flow of your system.

These are covered more extensively in our [events reference](site:reference/scheduling/events.md).

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
        if scheduler.running():
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
    print(f"Got exception {exception}")
    scheduler.stop()  # You can optionally pass `exception=` for logging purposes.

scheduler.run(on_exception="ignore")  # Scheduler will not stop because of the error
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

The second kind of exception that can happen is one that happens in the main process.
For example this could happen in one of your callbacks or in the `Scheduler` itself (please raise an issue if this occurs!).
By default when you call [`run()`][amltk.scheduling.Scheduler.run] it will set
`#!python run(on_exception="raise")` and raise the exception that occurred, with its traceback.
This is to help you debug your program.

You may also use `#!python run(on_exception="end")` which will just end the `Scheduler` and raise no exception,
or use `#!python run(on_exception="ignore")`, in which case the `Scheduler` will continue on with whatever events
are next to process.

## Tasks
Now that we have seen how the [`Scheduler`][amltk.scheduling.Scheduler] works,
we can look at the [`Task`][amltk.scheduling.Task], a wrapper around a function
that you'll want to submit to the `Scheduler`. The preferred way to create one
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
    collatz_task.submit(5)
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
    collatz_task.submit(5)

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
    echo_task.submit("hello")
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
one for the result of `#!python echo_task.submit("hello")` and the other
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
    task_1.submit(next_item)

@task_1.on_result
def submit_task_2_with_results_of_task_1(_, result: int) -> None:
    """When task_1 returns, send the result to task_2"""
    task_2.submit(result)

@task_1.on_result
def submit_task_1_with_next_item(_, result: int) -> None:
    """When task_1 returns, launch it again with the next items"""
    next_item = next(items, None)
    if next_item is not None:
        task_1.submit(next_item)
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
a [`Plugin`][amltk.scheduling.Plugin] to them. These plugins can automate control
behaviour of tasks, either through preventing their execution,
modifying the function and its arguments or even attaching plugin specific events!

For a complete reference, please see the [plugin reference page](site:reference/plugins).

### Call Limiter
Perhaps one of the more useful plugins, at least when designing an AutoML System is the
[`Limiter`][amltk.scheduling.plugins.Limiter] plugin. This can help you control
both it's concurrency or the absolute limit of how many times a certain task can be
successfully submitted.

In the following contrived example, we will setup a `Scheduler` with 2 workers and attempt
to submit a `Task` 4 times in rapid succession. However we have the constraint that we
only ever want 2 of these tasks running at a given time. Let's see how we could achieve that.

```python exec="true" source="material-block" html="True" hl_lines="9"
from amltk.scheduling import Scheduler, Limiter

def my_func(x: int) -> int:
    return x
from amltk._doc import make_picklable; make_picklable(my_func)  # markdown-exec: hide

scheduler = Scheduler.with_processes(2)

# Specify a concurrency limit of 2
task = scheduler.task(my_func, plugins=Limiter(max_concurrent=2))

# A list of 10 things we want to compute
items = iter(range(10))
results = []

@scheduler.on_start(repeat=4)  # Repeat callback 4 times
def submit() -> None:
    next_item = next(items)
    task.submit(next_item)

@task.on_result
def record_result(_, result: int) -> None:
    results.append(result)

@task.on_result
def launch_another(_, result: int) -> None:
    next_item = next(items, None)
    if next_item is not None:
        task.submit(next_item)

scheduler.run()
print(results)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

You can notice that this limiting worked, given the numbers `#!python 2` and `#!python 3`
were skipped and not printed. As expected, we successfully launched the task with both 
`#!python 0` and `#!python 1` but as these tasks were not done processing, the `Limiter`
kicks in and prevents the other two.

A natural extension to ask is then, "how do we requeue these?". Well lets take a look at the above
output. The plugin has added three new events to `Task`, namely
`@call-limit-reached`, `@concurrent-limit-reached` and `@disabled-due-to-running-task`.

To subscribe to these _extra_ events (or any for that matter), we can use
the [`task.on()`][amltk.scheduling.Task]
method. Below is the same example except here we respond to `@call-limit-reached`
and requeue the submissions that failed.

```python exec="true" source="material-block" html="True" hl_lines="11 19-21"
from amltk.scheduling import Scheduler, Limiter, Task
from amltk.types import Requeue

def my_func(x: int) -> int:
    return x
from amltk._doc import make_picklable; make_picklable(my_func)  # markdown-exec: hide

scheduler = Scheduler.with_processes(2)
task = scheduler.task(my_func, plugins=Limiter(max_concurrent=2))

# A list of 10 things we want to compute
items = Requeue(range(10))  # A convenience type that you can requeue/append to
results = []

@scheduler.on_start(repeat=4)  # Repeat callback 4 times
def submit() -> None:
    next_item = next(items)
    task.submit(next_item)

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
        task.submit(next_item)

scheduler.run()
print(results)
from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
```

### Under Construction

    Please see the following reference pages in the meantime:

    * [scheduler reference](site:reference/scheduling/scheduler.md) - A slighltly
        more condensed version of how to use the `Scheduler`.
    * [task reference](site:reference/scheduling/task.md) - A more comprehensive
        explanation of `Task`s and their `@events`.
    * [plugin reference](site:reference/scheduling/plugins.md) - An intro to plugins
        and how to create your own.
    * [executors reference](site:reference/scheduling/executors.md) - A list of
        executors and how to use them.
    * [events reference](site:reference/scheduling/events.md) - A more comprehensive
        look at the event system in AutoML-Toolkit and how to work with them or extend them.
