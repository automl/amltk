# Call Limiter
We can limit the number of times a function is called or how many concurrent
instances of it can be running. To do so, we create the [`CallLimiter`][amltk.CallLimiter]
and pass it in as a plugin. This plugin also introduces some new events that can be listened to.

```python
from amltk.scheduling import Task, CallLimiter

limiter = CallLimiter(max_calls=10, max_concurrent=2)
task = Task(..., plugins=[limiter])

@limiter.on_max_call_limit
def print_it(task: Task, *args, **kwargs) -> None:
    print(f"Task {task.name} was already called {task.n_called} times")

@limiter.on_max_concurrent
def print_it(task: Task, *args, **kwargs) -> None:
    print(f"Task {task.name} already running at max concurrency")
```

You can also prevent a task launching while another task is currently running:

```python
task1 = Task(...)

limiter = CallLimiter(not_while_running=task1)
task2 = Task(..., plugins=[limiter])

@limiter.on_disabled_due_to_running_task
def on_disabled_due_to_running_task(other_task: Task, task: Task, *args, **kwargs):
    print(
        f"Task {task.name} was not submitted because {other_task.name} is currently"
        " running"
    )
```
