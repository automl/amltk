# ThreadPoolCTL
Performing numerical operations in while multi-processing can create over-subscription
to threads by each process, especially when using numerical libraries like numpy,
scipy and sklearn. Specifically when training many sklearn models in different processes,
this can slow down training significantly with smaller datasets.

!!! note

    Plugin is only available if `threadpoolctl` is installed. You can so
    with `pip install amltk[threadpoolctl]`.

```python exec="true" source="material-block" result="python" title="ThreadPoolCTLPlugin example"
from amltk.scheduling import Scheduler
from amltk.threadpoolctl import ThreadPoolCTLPlugin

# Only used to verify, not needed if running
import threadpoolctl
import sklearn

print("------ Before")
print(threadpoolctl.threadpool_info())

scheduler = Scheduler.with_processes(1)

def f() -> None:
    print("------ Inside")
    print(threadpoolctl.threadpool_info())
from amltk._doc import make_picklable; make_picklable(f)  # markdown-exec: hide

task = scheduler.task(f, plugins=ThreadPoolCTLPlugin(max_threads=1))

@scheduler.on_start
def start_task() -> None:
    task()

scheduler.run()

print("------ After")
print(threadpoolctl.threadpool_info())
```
