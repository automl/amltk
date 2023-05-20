# Dask JobQueue
[`dask-jobqueue`](https://jobqueue.dask.org/en/latest/) is a package for
scheduling jobs across common clusters setups such as PBS, Slurm, MOAB,
SGE, LSF, and HTCondor.

You can access most of these directly through the _factory_ methods
of the [`Scheduler`][byop.Scheduler], forwarding on arguments to them.

!!! note "Factory Methods"

    * [`Scheduler.with_pbs()`][byop.scheduling.Scheduler.with_pbs]
    * [`Scheduler.with_lsf()`][byop.scheduling.Scheduler.with_lsf]
    * [`Scheduler.with_moab()`][byop.scheduling.Scheduler.with_moab]
    * [`Scheduler.with_sge()`][byop.scheduling.Scheduler.with_sge]
    * [`Scheduler.with_htcondor()`][byop.scheduling.Scheduler.with_htcondor]

Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
In particular, we only control the parameter `#!python n_workers=` to
use the [`adapt()`](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
method, every other keyword is forwarded to the relative
[cluster implementation](https://jobqueue.dask.org/en/latest/api.html).

In general, you should specify the requirements of each individual worker and
and tune your load with the `#!python n_workers=` parameter.

If you have any tips, tricks, working setups, gotchas, please feel free
to leave a PR or simply an issue!

=== "Slurm"

    ```python hl_lines="3 4 5 6 7 8 9"
    from byop.scheduling import Scheduler

    scheduler = Scheduler.with_slurm(
        n_workers=10,  # (1)!
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
       from byop.scheduling import Scheduler

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

    !!! warning "Running inside a job"

        Some cluster setups do not allow jobs to launch jobs themselves.
        The scheduler itself is lightweight and can run on the
        login node without issue. However you should make sure to offload
        heavy computations to a worker.

        If you get it to work, for example in an interactive job, please
        let us know!

    !!! info "Modifying the launch command"

        On some cluster commands, you'll need to modify the launch command.
        You can use the following to do so:

        ```python
        from byop.scheduling import Scheduler

        scheduler = Scheduler.with_slurm(n_workers=..., submit_command="sbatch --extra"
        ```

=== "Others"

    Please see the `dask-jobqueue` [documentation](https://jobqueue.dask.org/en/latest/)
    and the following methods:

