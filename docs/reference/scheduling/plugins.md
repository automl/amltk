## Plugins

Plugins are a way to modify a [`Task`][amltk.scheduling.task.Task], to add new functionality
or change the behaviour of what goes on in the function that is dispatched to the
[`Scheduler`][amltk.scheduling.Scheduler].

Some plugins will also add new `@event`s to a task, which can be used to respond accordingly to
something that may have occured with your task.

You can add a plugin to a [`Task`](site:reference/tasks/index.md) as so:

```python exec="true" html="true" source="material-block"
from amltk.scheduling import Task, Scheduler
from amltk.scheduling.plugins import Limiter

def some_function(x: int) -> int:
    return x * 2

scheduler = Scheduler.with_processes(1)

# When creating a task with the scheduler
task = scheduler.task(some_function, plugins=[Limiter(max_calls=10)])


# or directly to a Task
task = Task(some_function, scheduler=scheduler, plugins=[Limiter(max_calls=10)])
from amltk._doc import doc_print; doc_print(print, task)  # markdown-exec: hide
```

### Limiter
::: amltk.scheduling.plugins.limiter
    options:
        members: False

### Pynisher
::: amltk.scheduling.plugins.pynisher
    options:
        members: False

### Comm
::: amltk.scheduling.plugins.comm
    options:
        members: False

### ThreadPoolCTL
::: amltk.scheduling.plugins.threadpoolctl
    options:
        members: False

### Warning Filter
::: amltk.scheduling.plugins.warning_filter
    options:
        members: False

### Creating Your Own Plugin
::: amltk.scheduling.plugins.plugin
    options:
        members: False
