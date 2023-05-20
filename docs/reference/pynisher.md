# Pynisher
The plugin uses [pynisher](https://github.com/automl/pynisher) to place memory, cpu and walltime
constraints on processes, crashing them if these limits are reached.

It's best use is when used with [`Scheduler.with_processes()`][byop.Scheduler.with_processes] to have
work performed in processes.

??? warning "Scheduler Executor"

    This will place process limits on the task as soon as it starts
    running, whever it may be running. If you are using
    [`Scheduler.with_sequential()`][byop.Scheduler.with_sequential]
    then this will place limits on the main process, likely not what you
    want. This also does not work with a
    [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor].

    If using this with something like [`dask-jobqueue`](./dask-jobqueue.md),
    then this will place limits on the workers it spawns. It would be better
    to place limits directly through dask job-queue then.

??? warning "Platform Limitations"

    Pynisher has some limitations with memory on Mac and Windows:
    https://github.com/automl/pynisher#features


## Setting limits
To limit a task, we can create a [`PynisherPlugin`][byop.pynisher.PynisherPlugin] and
pass that to our [`Task`][byop.Task]. Each of the limits has an associated event
that can be listened to and acted upon if needed.

### Wall time

The maximum amount of wall clock time this task can use.
If the wall clock time limit triggered and the function crashes as a result,
the [`TIMEOUT`][byop.pynisher.PynisherPlugin.TIMEOUT] and
[`WALL_TIME_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.WALL_TIME_LIMIT_REACHED] events
will be emitted.

```python
from byop.scheduling import Task
from byop.pynisher import PynisherPlugin

pynisher = PynisherPlugin(wall_time_limit=(5, "m")) # (1)!
task = Task(..., plugins=[pynisher])

@task.on(pynisher.WALL_TIME_LIMIT_REACHED)
def print_it(exception):
    print(f"Failed with {exception=}")
```

1. Possible units are `#!python "s", "m", "h"`, defaults to `#!python "s"`

### Memory

The maximum amount of memory this task can use.
If the memory limit is triggered, the function crashes as a result, emitting the [`MEMORY_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.MEMORY_LIMIT_REACHED] event.

```python
from byop.scheduling import Task
from byop.pynisher import PynisherPlugin

pynisher = PynisherPlugin(memory_limit=(2, "gb")) # (1)!
task = Task(..., plugins=[pynisher])

@task.on(pynisher.MEMORY_LIMIT_REACHED)
def print_it(exception):
    print(f"Failed with {exception=}")
```

1. Possible units are `#!python "b", "kb", "mb", "gb"`, defaults to `#!python "b"`

!!! warning "Memory Limits with Pynisher"

    Pynisher has some limitations with memory on Mac and Windows:
    https://github.com/automl/pynisher#features

### CPU time

The maximum amount of CPU time this task can use.
If the CPU time limit triggered and the function crashes as a result,
the [`TIMEOUT`][byop.pynisher.PynisherPlugin.TIMEOUT] and
[`CPU_TIME_LIMIT_REACHED`][byop.pynisher.PynisherPlugin.CPU_TIME_LIMIT_REACHED]
events will be emitted.

```python
from byop.scheduling import Task
from byop.pynisher import PynisherPlugin

pynisher = PynisherPlugin(cpu_time_limit=(60, "s")) # (1)!
task = Task(..., plugins=[pynisher])

@task.on(pynisher.CPU_TIME_LIMIT_REACHED)
def print_it(exception):
    print(f"Failed with {exception=}")
```

1. Possible units are `#!python "s", "m", "h"`, defaults to `#!python "s"`

!!! warning "CPU Time Limits with Pynisher"

    Pynisher has some limitations with cpu timing on Mac and Windows:
    https://github.com/automl/pynisher#features
