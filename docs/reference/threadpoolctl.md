# ThreadPoolCTL
Performing numerical operations in while multi-processing can create over-subscription
to threads by each process, especially when using numerical libraries like numpy,
scipy and sklearn. Specifically when training many sklearn models in different processes,
this can slow down training significantly with smaller datasets.

!!! note

    Plugin is only available if `threadpoolctl` is installed. You can so
    with `pip install amltk[threadpoolctl]`.

```python exec="true" source="material-block" result="python" title="ThreadPoolCTLPlugin example"
from amltk.scheduling import Task, Scheduler
from amltk.threadpoolctl import ThreadPoolCTLPlugin

# Only used to verify, not needed if running
import threadpoolctl
import sklearn

print("------ Before")
print(threadpoolctl.threadpool_info())

scheduler = Scheduler.with_sequential()

def f() -> None:
    print("------ Inside")
    print(threadpoolctl.threadpool_info())

threadpoolctl_plugin = ThreadPoolCTLPlugin(max_threads=1)
task = Task(f, scheduler, plugins=[threadpoolctl_plugin])

@scheduler.on_start
def start_task() -> None:
    task()

scheduler.run()

print("------ After")
print(threadpoolctl.threadpool_info())
```
