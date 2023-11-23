## Executors

The [`Scheduler`][amltk.scheduling.Scheduler] uses
an [`Executor`][concurrent.futures.Executor], a builtin python native to
`#!python submit(f, *args, **kwargs)` to be computed
else where, whether it be locally or remotely.

```python
from amltk.scheduling import Scheduler

scheduler = Scheduler(executor=...)
```

Some parallelism libraries natively support this interface while we can
wrap others. You can also wrap you own custom backend by using
the `Executor` interface, which is relatively simple to implement.

If there's any executor background you wish to integrate, we would
be happy to consider it and greatly appreciate a PR!

### :material-language-python: `Python`

Python supports the `Executor` interface natively with the
[`concurrent.futures`][concurrent.futures] module for processes with the
[`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] and
[`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] for threads.

??? tip "Usage"

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

### :simple-dask: `dask`

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

### :simple-dask: `dask-jobqueue`

[`dask-jobqueue`](https://jobqueue.dask.org/en/latest/) is a package
for scheduling jobs across common clusters setups such as
PBS, Slurm, MOAB, SGE, LSF, and HTCondor.

Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
In particular, we only control the parameter `#!python n_workers=` and use `#!python adaptive=`
to control where to use [`adapt()`](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
or [`scale()`](https://jobqueue.dask.org/en/latest/howitworks.html?highlight=scale()#scheduler-and-jobs)
method, every other keyword is forwarded to the relative
[cluster implementation](https://jobqueue.dask.org/en/latest/api.html).

In general, you should specify the requirements of each individual worker
and tune your load with the `#!python n_workers=` parameter.

If you have any tips, tricks, working setups, gotchas, please feel free
to leave a PR or simply an issue!

??? tip "Usage"


    === "Slurm"

        ```python hl_lines="3 4 5 6 7 8 9"
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_slurm(
            n_workers=10,  # (1)!
            adaptive=True,
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

### :octicons-gear-24: `loky`

[Loky](https://loky.readthedocs.io/en/stable/API.html) is the default backend executor behind
[`joblib`](https://joblib.readthedocs.io/en/stable/), the parallelism that
powers scikit-learn.

??? tip "Usage"

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

??? warning "BLAS numeric backend"

    The loky executor seems to pick up on a different BLAS library (from scipy)
    which is different than those used by jobs from something like a `ProcessPoolExecutor`.

    This is likely not to matter for a majority of use-cases.

### :simple-ray: `ray`

[Ray](https://docs.ray.io/en/master/) is an open-source unified compute framework that makes it easy
to scale AI and Python workloads
— from reinforcement learning to deep learning to tuning,
and model serving.

!!! todo "In progress"

    Ray is currently in the works of supporting the Python
    `Executor` interface. See this [PR](https://github.com/ray-project/ray/pull/30826)
    for more info.

### :simple-apacheairflow: `airflow`

[Airflow](https://airflow.apache.org/) is a platform created by the community to programmatically author,
schedule and monitor workflows. Their list of integrations to platforms is endless
but features compute platforms such as Kubernetes, AWS, Microsoft Azure and
GCP.

!!! todo "In progress"

    We plan to support `airflow` in the future. If you'd like to help
    out, please reach out to us!

### :material-debug-step-over: Debugging

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
