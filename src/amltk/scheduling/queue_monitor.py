"""A monitor for the scheduler queue.

This module contains a monitor for the scheduler queue. The monitor tracks the
queue state at every event emitted by the scheduler. The data can be converted
to a pandas DataFrame or plotted as a stacked barchart.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from pandas.core.tools.datetimes import UnitChoices

    from amltk.scheduling import Scheduler


class QueueMonitorRecord(NamedTuple):
    """A record of the queue state at a given time."""

    time: int  # recorded in time_nanoseconds
    queue_size: int
    queued: int
    finished: int
    cancelled: int


class QueueMonitor:
    """A monitor for the scheduler queue.

    !!! note "Performance impact"

        This class attempts to be fairly lightweight, but it does add a bit of overhead
        as the monitor needs to be updated at every event emitted by the scheduler.
        While this likely won't impact

    May want to just integrate directly into the scheduler.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        """Initializes the monitor."""
        super().__init__()
        self.scheduler = scheduler
        self.data: list[QueueMonitorRecord] = []

        scheduler.on_start(self.update)
        scheduler.on_finishing(self.update)
        scheduler.on_finished(self.update)
        scheduler.on_future_submitted(self.update)
        scheduler.on_future_cancelled(self.update)
        scheduler.on_future_done(self.update)

    def df(
        self,
        *,
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """Converts the data to a pandas DataFrame.

        Args:
            n_workers: The number of workers that were in use. This helps idenify how
                many workers were idle at a given time. If None, the maximum length of
                the queue at any recorded time is used.
        """
        _df = pd.DataFrame(self.data, columns=list(QueueMonitorRecord._fields)).astype(
            {
                # Windows might have a weird default here but it should be 64 at least
                "time": "int64",
                "queue_size": int,
                "queued": int,
                "finished": int,
                "cancelled": int,
            },
        )
        if n_workers is None:
            n_workers = int(_df["queue_size"].max())
        _df["idle"] = n_workers - _df["queue_size"]
        _df["time"] = pd.to_datetime(_df["time"], unit="ns", origin="unix")
        return _df.set_index("time")

    def plot(
        self,
        *,
        ax: plt.Axes | None = None,
        interval: tuple[int, UnitChoices] = (1, "s"),
        n_workers: int | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plots the data as a stacked barchart.

        Args:
            ax: The axes to plot on. If None, a new figure is created.
            interval: The interval to use for the x-axis. The first value is the
                interval and the second value is the unit. Must be a valid pandas
                timedelta unit. See [to_timedelta()][pandas.to_timedelta] for more
                information.
            n_workers: The number of workers that were in use. This helps idenify how
                many workers were idle at a given time. If None, the maximum length of
                the queue at any recorded time is used.
            **kwargs: Additional keyword arguments to pass to the pandas plot function.

        Returns:
            The axes.
        """
        if ax is None:
            _, _ax = plt.subplots(1, 1)
        else:
            _ax = ax

        _df = self.df(n_workers=n_workers)
        _df = _df.resample(f"{interval[0]}{interval[1]}").mean()
        _df.index = _df.index - _df.index[0]
        _reversed_df = _df[::-1]

        _reversed_df.plot.barh(
            stacked=True,
            y=["finished", "queued", "cancelled", "idle"],
            ax=_ax,
            width=1,
            edgecolor="k",
            **kwargs,
        )

        _ax.set_ylabel("Time")
        _ax.yaxis.set_major_locator(MaxNLocator(nbins="auto"))

        _ax.set_xlabel("Count")
        _ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        return _ax

    def update(self, *_: Any) -> None:
        """Updates the data when the scheduler has an event."""
        queue = self.scheduler.queue
        # OPTIM: Not sure if this is fastenough
        counter = Counter([f._state for f in queue])
        self.data.append(
            QueueMonitorRecord(
                time.time_ns(),
                len(queue),
                counter["PENDING"],
                counter["FINISHED"],
                counter["CANCELLED"],
            ),
        )
