## Profiling
Whether for debugging, building an AutoML system or for optimization
purposes, we provide a powerful [`Profiler`][amltk.profiling.Profiler],
which can generate a [`Profile`][amltk.profiling.Profile] of different sections
of code. This is particularly useful with [`Trial`][amltk.optimization.Trial]s,
so much so that we attach one to every `Trial` made as
[`trial.profiler`][amltk.optimization.Trial.profiler].

When done profiling, you can export all generated profiles as a dataframe using
[`profiler.df()`][amltk.profiling.Profiler.df].

```python exec="true" result="python" source="material-block"
from amltk.profiling import Profiler
import numpy as np

profiler = Profiler()

with profiler("loading-data"):
    X = np.random.rand(1000, 1000)

with profiler("training-model"):
    model = np.linalg.inv(X)

with profiler("predicting"):
    y = model @ X

print(profiler.df())
```

You'll find these profiles as keys in the [`Profiler`][amltk.profiling.Profiler],
e.g. `#! python profiler["loading-data"]`.

This will measure both the time it took within the block but also
the memory consumed before and after the block finishes, allowing
you to get an estimate of the memory consumed.


??? tip "Memory, vms vs rms"

    While not entirely accurate, this should be enough for info
    for most use cases.

    Given the main process uses 2GB of memory and the process
    then spawns a new process in which you are profiling, as you
    might do from a [`Task`][amltk.scheduling.Task]. In this new
    process you use another 2GB on top of that, then:

    * The virtual memory size (**vms**) will show 4GB as the
    new process will share the 2GB with the main process and
    have it's own 2GB.

    * The resident set size (**rss**) will show 2GB as the
    new process will only have 2GB of it's own memory.


If you need to profile some iterator, like a for loop, you can use
[`Profiler.each()`][amltk.profiling.Profiler.each] which will measure
the entire loop but also each individual iteration. This can be useful
for iterating batches of a deep-learning model, splits of a cross-validator
or really any loop with work you want to profile.

```python exec="true" result="python" source="material-block"
from amltk.profiling import Profiler
import numpy as np

profiler = Profiler()

for i in profiler.each(range(3), name="for-loop"):
    X = np.random.rand(1000, 1000)

print(profiler.df())
```

Lastly, to disable profiling without editing much code,
you can always use [`Profiler.disable()`][amltk.profiling.Profiler.disable]
and [`Profiler.enable()`][amltk.profiling.Profiler.enable] to toggle
profiling on and off.
