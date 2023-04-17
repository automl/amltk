"""Dask Jobqueue Executors.

These are essentially wrappers around the dask_jobqueue classes.
We use them to provide a consistent interface for all the different
jobqueue implementations and get access to their executors.

!!! example "Documentation from `dask_jobqueue`"

    See the [dask jobqueue documentation](https://jobqueue.dask.org/en/latest/) specifically:

    * [Example deployments](https://jobqueue.dask.org/en/latest/examples.html#example-deployments)
    * [Tips and Tricks](https://jobqueue.dask.org/en/latest/advanced-tips-and-tricks.html)
    * [Debugging](https://jobqueue.dask.org/en/latest/debug.html)

"""
# ruff: noqa: N802, E501
from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
)
from typing_extensions import ParamSpec, TypeAlias

from dask_jobqueue import (
    HTCondorCluster,
    JobQueueCluster,
    LSFCluster,
    MoabCluster,
    OARCluster,
    PBSCluster,
    SGECluster,
    SLURMCluster,
)

if TYPE_CHECKING:
    from distributed.cfexecutor import ClientExecutor

R = TypeVar("R")
P = ParamSpec("P")
_JQC = TypeVar("_JQC", bound=JobQueueCluster)

DJQ_NAMES: TypeAlias = Literal["slurm", "htcondor", "lsf", "oar", "pbs", "sge", "moab"]


class DaskJobqueueExecutor(Executor, Generic[_JQC]):
    """A concurrent.futures Executor that executes tasks on a dask_jobqueue cluster."""

    def __init__(self, cluster: _JQC, n_workers: int):
        """Initialize a DaskJobqueueExecutor.

        This will specifically use the [dask_jobqueue.JobQueueCluster.adapt][] method to
        dynamically scale the cluster to the number of workers specified.

        !!! note "Implementations"

            Prefer to use the class methods to create an instance of this class.

            * [`DaskJobqueueExecutor.SLURM()`][byop.dask_jobqueue.DaskJobqueueExecutor.SLURM]
            * [`DaskJobqueueExecutor.HTCondor()`][byop.dask_jobqueue.DaskJobqueueExecutor.HTCondor]
            * [`DaskJobqueueExecutor.LSF()`][byop.dask_jobqueue.DaskJobqueueExecutor.LSF]
            * [`DaskJobqueueExecutor.OAR()`][byop.dask_jobqueue.DaskJobqueueExecutor.OAR]
            * [`DaskJobqueueExecutor.PBS()`][byop.dask_jobqueue.DaskJobqueueExecutor.PBS]
            * [`DaskJobqueueExecutor.SGE()`][byop.dask_jobqueue.DaskJobqueueExecutor.SGE]
            * [`DaskJobqueueExecutor.Moab()`][byop.dask_jobqueue.DaskJobqueueExecutor.Moab]

        Args:
            cluster: The implementation of a [dask_jobqueue.JobQueueCluster][].
            n_workers: The number of workers to maximally adapt to on the cluster.
        """
        self.cluster = cluster
        self.cluster.adapt(minimum=0, maximum=n_workers)
        self.n_workers = n_workers
        self.executor: ClientExecutor = self.cluster.get_client().get_executor()

    def submit(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        """See [concurrent.futures.Executor.submit][]."""
        future = self.executor.submit(fn, *args, **kwargs)
        assert isinstance(future, Future)
        return future

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable,
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[R]:
        """See [concurrent.futures.Executor.map][]."""
        return self.executor.map(  # type: ignore
            fn,
            *iterables,
            timeout=timeout,
            chunksize=chunksize,
        )

    def shutdown(
        self,
        wait: bool = True,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> None:
        """See [concurrent.futures.Executor.shutdown][]."""
        self.executor.shutdown(wait=wait, **kwargs)

    @classmethod
    def SLURM(
        cls,
        *,
        n_workers: int,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[SLURMCluster]:
        """Create a DaskJobqueueExecutor for a SLURM cluster.

        See the [dask_jobqueue.SLURMCluster documentation][dask_jobqueue.SLURMCluster] for
        more information on the available keyword arguments.
        """
        return cls(SLURMCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def HTCondor(
        cls,
        *,
        n_workers: int,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[HTCondorCluster]:
        """Create a DaskJobqueueExecutor for a HTCondor cluster.

        See the [dask_jobqueue.HTCondorCluster documentation][dask_jobqueue.HTCondorCluster] for
        more information on the available keyword arguments.
        """
        return cls(HTCondorCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def LSF(cls, *, n_workers: int, **kwargs: Any) -> DaskJobqueueExecutor[LSFCluster]:
        """Create a DaskJobqueueExecutor for a LSF cluster.

        See the [dask_jobqueue.LSFCluster documentation][dask_jobqueue.LSFCluster] for
        more information on the available keyword arguments.
        """
        return cls(LSFCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def OAR(cls, *, n_workers: int, **kwargs: Any) -> DaskJobqueueExecutor[OARCluster]:
        """Create a DaskJobqueueExecutor for a OAR cluster.

        See the [dask_jobqueue.OARCluster documentation][dask_jobqueue.OARCluster] for
        more information on the available keyword arguments.
        """
        return cls(OARCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def PBS(cls, *, n_workers: int, **kwargs: Any) -> DaskJobqueueExecutor[PBSCluster]:
        """Create a DaskJobqueueExecutor for a PBS cluster.

        See the [dask_jobqueue.PBSCluster documentation][dask_jobqueue.PBSCluster] for
        more information on the available keyword arguments.
        """
        return cls(PBSCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def SGE(cls, *, n_workers: int, **kwargs: Any) -> DaskJobqueueExecutor[SGECluster]:
        """Create a DaskJobqueueExecutor for a SGE cluster.

        See the [dask_jobqueue.SGECluster documentation][dask_jobqueue.SGECluster] for
        more information on the available keyword arguments.
        """
        return cls(SGECluster(**kwargs), n_workers=n_workers)

    @classmethod
    def Moab(
        cls,
        *,
        n_workers: int,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[MoabCluster]:
        """Create a DaskJobqueueExecutor for a Moab cluster.

        See the [dask_jobqueue.MoabCluster documentation][dask_jobqueue.MoabCluster] for
        more information on the available keyword arguments.
        """
        return cls(MoabCluster(**kwargs), n_workers=n_workers)

    @classmethod
    def from_str(
        cls,
        name: DJQ_NAMES,
        *,
        n_workers: int,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor:
        """Create a DaskJobqueueExecutor using a string lookup.

        Args:
            name: The name of cluster to create, must be one of
                ["slurm", "htcondor", "lsf", "oar", "pbs", "sge", "moab"].
            n_workers: The number of workers to maximally adapt to on the cluster.
            kwargs: The keyword arguments to pass to the cluster constructor.

        Raises:
            KeyError: If `name` is not one of the supported cluster types.

        Returns:
            A DaskJobqueueExecutor for the requested cluster type.
        """
        methods = {
            "slurm": cls.SLURM,
            "htcondor": cls.HTCondor,
            "lsf": cls.LSF,
            "oar": cls.OAR,
            "pbs": cls.PBS,
            "sge": cls.SGE,
            "moab": cls.Moab,
        }
        method = methods.get(name.lower())
        if method is None:
            raise KeyError(
                f"Unknown cluster name: {name}, must be from {list(methods)}",
            )

        return method(n_workers=n_workers, **kwargs)
