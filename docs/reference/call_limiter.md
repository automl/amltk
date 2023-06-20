# Call Limiter
We can limit the number of times a function is called or how many concurrent
instances of it can be running. To do so, we create the [`CallLimiter`][amltk.CallLimiter]
and pass it in as a plugin. This plugin also introduces some new events that can be listened to.

```python
from amltk.scheduling import Task, CallLimiter

limiter = CallLimiter(max_calls=10, max_concurrent=2)
task = Task(..., plugins=[limiter])

@task.on(CallLimiter.CALL_LIMIT_REACHED)
def print_it(*args, **kwargs) -> None:
    print(f"Task was already called {task.n_called} times")

@task.on(CallLimiter.CONCURRENT_LIMIT_REACHED)
def print_it(*args, **kwargs) -> None:
    print(f"Task already running at max concurrency")
```

!!! note "Events"

    * [`CALL_LIMIT_REACHED`][amltk.CallLimiter.CALL_LIMIT_REACHED]
        Emitted when the call limit is reached

    * [`CONCURRENT_LIMIT_REACHED`][amltk.CallLimiter.CONCURRENT_LIMIT_REACHED]
        Emitted when the concurrent task limit is reached.
