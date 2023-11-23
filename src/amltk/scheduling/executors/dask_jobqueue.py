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

import logging
import pprint
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar
from typing_extensions import ParamSpec, Self, override

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

logger = logging.getLogger(__name__)


class DaskJobqueueExecutor(Executor, Generic[_JQC]):
    """A concurrent.futures Executor that executes tasks on a dask_jobqueue cluster."""

    def __init__(
        self,
        cluster: _JQC,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
    ):
        """Initialize a DaskJobqueueExecutor.


        !!! note "Implementations"

            Prefer to use the class methods to create an instance of this class.

            * [`DaskJobqueueExecutor.SLURM()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.SLURM]
            * [`DaskJobqueueExecutor.HTCondor()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.HTCondor]
            * [`DaskJobqueueExecutor.LSF()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.LSF]
            * [`DaskJobqueueExecutor.OAR()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.OAR]
            * [`DaskJobqueueExecutor.PBS()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.PBS]
            * [`DaskJobqueueExecutor.SGE()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.SGE]
            * [`DaskJobqueueExecutor.Moab()`][amltk.scheduling.executors.dask_jobqueue.DaskJobqueueExecutor.Moab]

        Args:
            cluster: The implementation of a
                [dask_jobqueue.JobQueueCluster](https://jobqueue.dask.org/en/latest/api.html).
            n_workers: The number of workers to maximally adapt to on the cluster.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed allocate all workers.
                This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers specified.
            submit_command: To overwrite the submission command if necessary.
            cancel_command: To overwrite the cancel command if necessary.
        """
        super().__init__()
        self.cluster = cluster
        self.adaptive = adaptive
        if submit_command:
            self.cluster.job_cls.submit_command = submit_command  # type: ignore

        if cancel_command:
            self.cluster.job_cls.cancel_command = cancel_command  # type: ignore

        if adaptive:
            self.cluster.adapt(minimum=0, maximum=n_workers)
        else:
            self.cluster.scale(n_workers)

        self.n_workers = n_workers
        self.executor: ClientExecutor = self.cluster.get_client().get_executor()

    @override
    def __enter__(self) -> Self:
        configuration = {
            "header": self.cluster.job_header,
            "script": self.cluster.job_script(),
            "job_name": self.cluster.job_name,
        }
        config_str = pprint.pformat(configuration)
        logger.debug(f"Launching script with configuration:\n {config_str}")
        self.executor.__enter__()
        return self

    @override
    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.executor.__exit__(*args, **kwargs)

    @override
    def submit(
        self,
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[R]:
        """See [concurrent.futures.Executor.submit][]."""
        future = self.executor.submit(fn, *args, **kwargs)
        assert isinstance(future, Future)
        return future

    @override
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

    @override
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
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[SLURMCluster]:
        """Create a DaskJobqueueExecutor for a SLURM cluster.

        See the [dask_jobqueue.SLURMCluster documentation][dask_jobqueue.SLURMCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            SLURMCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            n_workers=n_workers,
            adaptive=adaptive,
        )

    @classmethod
    def HTCondor(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[HTCondorCluster]:
        """Create a DaskJobqueueExecutor for a HTCondor cluster.

        See the [dask_jobqueue.HTCondorCluster documentation][dask_jobqueue.HTCondorCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            HTCondorCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def LSF(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[LSFCluster]:
        """Create a DaskJobqueueExecutor for a LSF cluster.

        See the [dask_jobqueue.LSFCluster documentation][dask_jobqueue.LSFCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            LSFCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def OAR(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[OARCluster]:
        """Create a DaskJobqueueExecutor for a OAR cluster.

        See the [dask_jobqueue.OARCluster documentation][dask_jobqueue.OARCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            OARCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def PBS(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[PBSCluster]:
        """Create a DaskJobqueueExecutor for a PBS cluster.

        See the [dask_jobqueue.PBSCluster documentation][dask_jobqueue.PBSCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            PBSCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def SGE(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[SGECluster]:
        """Create a DaskJobqueueExecutor for a SGE cluster.

        See the [dask_jobqueue.SGECluster documentation][dask_jobqueue.SGECluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            SGECluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def Moab(
        cls,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor[MoabCluster]:
        """Create a DaskJobqueueExecutor for a Moab cluster.

        See the [dask_jobqueue.MoabCluster documentation][dask_jobqueue.MoabCluster] for
        more information on the available keyword arguments.
        """
        return cls(  # type: ignore
            MoabCluster(**kwargs),
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            n_workers=n_workers,
        )

    @classmethod
    def from_str(
        cls,
        name: DJQ_NAMES,
        *,
        n_workers: int,
        adaptive: bool = False,
        submit_command: str | None = None,
        cancel_command: str | None = None,
        **kwargs: Any,
    ) -> DaskJobqueueExecutor:
        """Create a DaskJobqueueExecutor using a string lookup.

        Args:
            name: The name of cluster to create, must be one of
                ["slurm", "htcondor", "lsf", "oar", "pbs", "sge", "moab"].
            n_workers: The number of workers to maximally adapt to on the cluster.
            adaptive: Whether to use the adaptive scaling of the cluster or fixed allocate all workers.
                This will specifically use the
                [dask_jobqueue.SLURMCluster.adapt](https://jobqueue.dask.org/en/latest/index.html?highlight=adapt#adaptivity)
                method to dynamically scale the cluster to the number of workers specified.
            submit_command: Overwrite the submit command of workers if necessary.
            cancel_command: Overwrite the cancel command of workers if necessary.
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

        return method(
            n_workers=n_workers,
            submit_command=submit_command,
            cancel_command=cancel_command,
            adaptive=adaptive,
            **kwargs,
        )
