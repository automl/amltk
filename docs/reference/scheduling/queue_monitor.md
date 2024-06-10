## Queue Monitor
A [`QueueMonitor`][amltk.scheduling.queue_monitor.QueueMonitor] is a
monitor for the scheduler queue.

This module contains a monitor for the scheduler queue. The monitor tracks the
queue state at every event emitted by the scheduler. The data can be converted
to a pandas DataFrame or plotted as a stacked barchart.

!!! note "Monitoring Frequency"

    To prevent repeated polling, we sample the scheduler queue at every scheduler event.
    This is because the queue is only modified upon one of these events. This means we
    don't need to poll the queue at a fixed interval. However, if you need more fine
    grained updates, you can add extra events/timings at which the monitor should
    [`update()`][amltk.scheduling.queue_monitor.QueueMonitor.update].

!!! warning "Performance impact"

    If your tasks and callbacks are very fast (~sub 10ms), then the monitor has a
    non-nelgible impact however for most use cases, this should not be a problem.
    As anything, you should profile how much work the scheduler can get done,
    with and without the monitor, to see if it is a problem for your use case.

In the below example, we have a very fast running function that runs on repeat,
sometimes too fast for the scheduler to keep up, letting some futures buildup needing
to be processed.

```python exec="true" source="material-block" result="python" session="queue-monitor"
import time
import matplotlib.pyplot as plt
from amltk.scheduling import Scheduler
from amltk.scheduling.queue_monitor import QueueMonitor

def fast_function(x: int) -> int:
    return x + 1
from amltk._doc import make_picklable; make_picklable(fast_function)  # markdown-exec: hide

N_WORKERS = 2
scheduler = Scheduler.with_processes(N_WORKERS)
monitor = QueueMonitor(scheduler)
task = scheduler.task(fast_function)

@scheduler.on_start(repeat=N_WORKERS)
def start():
    task.submit(1)

@task.on_result
def result(_, x: int):
    if scheduler.running():
        task.submit(x)

scheduler.run(timeout=1)
df = monitor.df()
print(df)
```

We can also [`plot()`][amltk.scheduling.queue_monitor.QueueMonitor.plot] the data as a
stacked barchart with a set interval.

```python exec="true" source="material-block" html="true" session="queue-monitor"
fig, ax = plt.subplots()
monitor.plot(interval=(50, "ms"))
from io import StringIO; fig.tight_layout(); buffer = StringIO(); plt.savefig(buffer, format="svg"); print(buffer.getvalue())  # markdown-exec: hide
```
