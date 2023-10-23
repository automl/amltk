"""A trial for an optimization task.

TODO: Populate more here.
"""
from __future__ import annotations

import copy
import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    overload,
)
from typing_extensions import ParamSpec, Self, override

import numpy as np
import pandas as pd

from amltk.functional import dict_get_not_none, mapping_select, prefix_keys
from amltk.profiling import Memory, Profile, Profiler, Timer
from amltk.store import Bucket, PathBucket

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.panel import Panel
    from rich.text import Text

# Inner trial info object
I = TypeVar("I")  # noqa: E741
I2 = TypeVar("I2")

# Parameters to task
P = ParamSpec("P")
P2 = ParamSpec("P2")

# Return type of task
R = TypeVar("R")

# Random TypeVar
T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class Trial(Generic[I]):
    """A trial as suggested by an optimizer.

    You can modify the Trial as you see fit, specifically
    `.summary` which are for recording any information you may like.

    The other attributes will be automatically set, such
    as `.profile` and `.exception`, which are capture
    using [`trial.begin()`][amltk.optimization.trial.Trial.begin].

    Args:
        name: The unique name of the trial.
        config: The config for the trial.
        info: The info of the trial.
        seed: The seed to use if suggested by the optimizer.
    """

    name: str
    """The unique name of the trial."""

    config: Mapping[str, Any]
    """The config of the trial provided by the optimizer."""

    info: I = field(repr=False)
    """The info of the trial provided by the optimizer."""

    seed: int | None = None
    """The seed to use if suggested by the optimizer."""

    fidelities: dict[str, Any] | None = None
    """The fidelities at which to evaluate the trial, if any."""

    time: Timer.Interval = field(repr=False, default_factory=Timer.na)
    """The time taken by the trial, once ended."""

    memory: Memory.Interval = field(repr=False, default_factory=Memory.na)
    """The memory used by the trial, once ended."""

    profiler: Profiler = field(
        repr=False,
        default_factory=lambda: Profiler(memory_unit="B", time_kind="wall"),
    )
    """A profiler for this trial."""

    summary: dict[str, Any] = field(default_factory=dict)
    """The summary of the trial. These are for summary statistics of a trial and
    are single values."""

    results: dict[str, Any] = field(default_factory=dict)
    """The results of the trial. These are what will be reported to the optimizer.
    These are mainly set by the [`success()`][amltk.optimization.trial.Trial.success]
    and [`fail()`][amltk.optimization.trial.Trial.fail] methods."""

    exception: BaseException | None = field(repr=True, default=None)
    """The exception raised by the trial, if any."""

    traceback: str | None = field(repr=False, default=None)
    """The traceback of the exception, if any."""

    storage: set[Any] = field(default_factory=set)
    """Anything stored in the trial, the elements of the list are keys that can be
    used to retrieve them later, such as a Path.
    """

    plugins: dict[str, Any] = field(default_factory=dict)
    """Any plugins attached to the trial."""

    @property
    def profiles(self) -> Mapping[str, Profile.Interval]:
        """The profiles of the trial."""
        return self.profiler.profiles

    @contextmanager
    def begin(
        self,
        time: Timer.Kind | Literal["wall", "cpu", "process"] | None = None,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] | None = None,
    ) -> Iterator[None]:
        """Begin the trial with a `contextmanager`.

        Will begin timing the trial in the `with` block, attaching the profiled time and memory
        to the trial once completed, under `.profile.time` and `.profile.memory` attributes.

        If an exception is raised, it will be attached to the trial under `.exception`
        with the traceback attached to the actual error message, such that it can
        be pickled and sent back to the main process loop.

        ```python exec="true" source="material-block" result="python" title="begin" hl_lines="5"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.begin():
            # Do some work
            pass

        print(trial.memory)
        print(trial.time)
        ```

        ```python exec="true" source="material-block" result="python" title="begin-fail" hl_lines="5"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": -1}, info={})

        with trial.begin():
            raise ValueError("x must be positive")

        print(trial.exception)
        print(trial.traceback)
        print(trial.memory)
        print(trial.time)
        ```

        Args:
            time: The timer kind to use for the trial. Defaults to the default
                timer kind of the profiler.
            memory_unit: The memory unit to use for the trial. Defaults to the
                default memory unit of the profiler.
        """  # noqa: E501
        with self.profiler(name="trial", memory_unit=memory_unit, time_kind=time):
            try:
                yield
            except Exception as error:  # noqa: BLE001
                self.exception = error
                self.traceback = traceback.format_exc()
            finally:
                self.time = self.profiler["trial"].time
                self.memory = self.profiler["trial"].memory

    @contextmanager
    def profile(
        self,
        name: str,
        *,
        time: Timer.Kind | Literal["wall", "cpu", "process"] | None = None,
        memory_unit: Memory.Unit | Literal["B", "KB", "MB", "GB"] | None = None,
        summary: bool = False,
    ) -> Iterator[None]:
        """Measure some interval in the trial.

        The results of the profiling will be available in the `.summary` attribute
        with the name of the interval as the key.

        ```python exec="true" source="material-block" result="python" title="profile"
        from amltk.optimization import Trial
        import time

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.profile("some_interval"):
            # Do some work
            time.sleep(1)

        print(trial.profiler["some_interval"].time)
        ```

        Args:
            name: The name of the interval.
            time: The timer kind to use for the trial. Defaults to the default
                timer kind of the profiler.
            memory_unit: The memory unit to use for the trial. Defaults to the
                default memory unit of the profiler.
            summary: Whether to add the interval to the summary.

        Yields:
            The interval measured. Values will be nan until the with block is finished.
        """
        with self.profiler(name=name, memory_unit=memory_unit, time_kind=time):
            yield

        if summary:
            profile = self.profiler[name]
            self.summary.update(profile.to_dict(prefix=name))

    def success(self, **results: Any) -> Trial.Report[I]:
        """Generate a success report.

        ```python exec="true" source="material-block" result="python" title="success" hl_lines="7"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.begin():
            # Do some work
            report = trial.success(cost=1)

        print(report)
        ```

        Returns:
            The result of the trial.
        """  # noqa: E501
        self.results = results
        return Trial.Report(trial=self, status=Trial.Status.SUCCESS)

    def fail(self, **results: Any) -> Trial.Report[I]:
        """Generate a failure report.

        ```python exec="true" source="material-block" result="python" title="fail" hl_lines="6 9 10"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        with trial.begin():
            raise ValueError("This is an error")  # Something went wrong
            report = trial.success(cost=1)

        if trial.exception: # You can check for an exception of the trial here
            report = trial.fail(cost=100)

        print(report)
        ```

        Returns:
            The result of the trial.
        """  # noqa: E501
        self.results = results
        return Trial.Report(trial=self, status=Trial.Status.FAIL)

    def crashed(
        self,
        exception: BaseException | None = None,
        traceback: str | None = None,
    ) -> Trial.Report[I]:
        """Generate a crash report.

        !!! note

            You will typically not create these manually, but instead if we don't
            recieve a report from a target function evaluation, but only an error,
            we assume something crashed and generate a crash report for you.

        Args:
            exception: The exception that caused the crash. If not provided, the
                exception will be taken from the trial. If this is still `None`,
                a `RuntimeError` will be raised.
            traceback: The traceback of the exception. If not provided, the
                traceback will be taken from the trial if there is one there.

        Returns:
            The result of the trial.
        """
        if exception is None and self.exception is None:
            raise RuntimeError(
                "Cannot generate a crash report without an exception."
                " Please provide an exception or use `with trial.begin():` to start"
                " the trial.",
            )

        self.exception = exception if exception else self.exception
        self.traceback = traceback if traceback else self.traceback

        return Trial.Report(trial=self, status=Trial.Status.CRASHED)

    def store(
        self,
        items: Mapping[str, T],
        *,
        where: str | Path | Bucket | Callable[[str, Mapping[str, T]], None],
    ) -> None:
        """Store items related to the trial.

        ```python exec="true" source="material-block" result="python" title="store" hl_lines="5"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        print(trial.storage)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="store-bucket" hl_lines="8"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where=bucket)

        print(trial.storage)
        ```

        Args:
            items: The items to store, a dict from the key to store it under
                to the item itself. If using a `str`, `Path` or `PathBucket`,
                the keys of the items should be a valid filename, including
                the correct extension. e.g. `#!python {"config.json": trial.config}`

            where: Where to store the items.

                * If a `str` or `Path`, will store
                a bucket will be created at the path, and the items will be
                stored in a sub-bucket with the name of the trial.

                * If a `Bucket`, will store the items in a sub-bucket with the
                name of the trial.

                * If a `Callable`, will call the callable with the name of the
                trial and the key-valued pair of items to store.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        method: Callable[[str, dict[str, Any]], None] | Bucket
        if isinstance(where, str):
            where = Path(where)
            method = PathBucket(where, create=True)
        elif isinstance(where, Path):
            method = PathBucket(where, create=True)
        else:
            method = where

        if isinstance(method, Bucket):
            # Store in a sub-bucket
            method.sub(self.name).store(items)
        else:
            # Leave it up to supplied method
            method(self.name, items)

        # Add the keys to storage
        self.storage.update(items.keys())

    def delete_from_storage(
        self,
        items: Iterable[str],
        *,
        where: str | Path | Bucket | Callable[[str, Iterable[str]], dict[str, bool]],
    ) -> dict[str, bool]:
        """Delete items related to the trial.

        ```python exec="true" source="material-block" result="python" title="delete-storage" hl_lines="6"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")
        trial.delete_from_storage(items=["config.json"], where="./results")

        print(trial.storage)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="delete-storage-bucket" hl_lines="9"
        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where=bucket)
        trial.delete_from_storage(items=["config.json"], where=bucket)

        print(trial.storage)
        ```

        Args:
            items: The items to delete, an iterable of keys
            where: Where the items are stored

                * If a `str` or `Path`, will lookup a bucket at the path,
                and the items will be deleted from a sub-bucket with the name of the trial.

                * If a `Bucket`, will delete the items in a sub-bucket with the
                name of the trial.

                * If a `Callable`, will call the callable with the name of the
                trial and the keys of the items to delete. Should a mapping from
                the key to whether it was deleted or not.

        Returns:
            A dict from the key to whether it was deleted or not.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        method: Bucket
        if isinstance(where, str):
            method = PathBucket(Path(where), create=False)
        elif isinstance(where, Path):
            method = PathBucket(where, create=False)
        elif isinstance(where, Bucket):
            method = where
        else:
            # Leave it up to supplied method
            return where(self.name, items)

        sub_bucket = method.sub(self.name)
        return sub_bucket.remove(items)

    def copy(self) -> Self:
        """Create a copy of the trial.

        Returns:
            The copy of the trial.
        """
        return copy.deepcopy(self)

    @overload
    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: None = None,
    ) -> Any:
        ...

    @overload
    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: type[R],
    ) -> R:
        ...

    def retrieve(
        self,
        key: str,
        *,
        where: str | Path | Bucket[str, Any],
        check: type[R] | None = None,
    ) -> R | Any:
        """Retrieve items related to the trial.

        !!! note "Same argument for `where=`"

             Use the same argument for `where=` as you did for `store()`.

        ```python exec="true" source="material-block" result="python" title="retrieve" hl_lines="7"
        from amltk.optimization import Trial

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where="./results")

        config = trial.retrieve("config.json", where="./results")
        print(config)
        ```

        You could also create a Bucket and use that instead.

        ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

        from amltk.optimization import Trial
        from amltk.store import PathBucket

        bucket = PathBucket("results")

        trial = Trial(name="trial", config={"x": 1}, info={})

        trial.store({"config.json": trial.config}, where=bucket)

        config = trial.retrieve("config.json", where=bucket)
        print(config)
        ```

        Args:
            key: The key of the item to retrieve as said in `.storage`.
            check: If provided, will check that the retrieved item is of the
                provided type. If not, will raise a `TypeError`. This
                is only used if `where=` is a `str`, `Path` or `Bucket`.
            where: Where to retrieve the items from.

                * If a `str` or `Path`, will store
                a bucket will be created at the path, and the items will be
                retrieved from a sub-bucket with the name of the trial.

                * If a `Bucket`, will retrieve the items from a sub-bucket with the
                name of the trial.

        Returns:
            The retrieved item.

        Raises:
            TypeError: If `check=` is provided and  the retrieved item is not of the provided
                type.
        """  # noqa: E501
        # If not a Callable, we convert to a path bucket
        method: Bucket[str, Any]
        if isinstance(where, str):
            where = Path(where)
            method = PathBucket(where, create=True)
        elif isinstance(where, Path):
            method = PathBucket(where, create=True)
        else:
            method = where

        # Store in a sub-bucket
        return method.sub(self.name)[key].load(check=check)

    def attach_plugin_item(self, name: str, plugin_item: Any) -> None:
        """Attach a plugin item to the trial.

        Args:
            name: The name of the plugin item.
            plugin_item: The plugin item.
        """
        self.plugins[name] = plugin_item

    def rich_renderables(self) -> Iterable[RenderableType]:
        """The renderables for rich for this report."""
        from rich.panel import Panel
        from rich.pretty import Pretty

        if self.exception:
            yield Panel(Pretty(self.exception), title="Exception", title_align="left")

        yield Panel(Pretty(self.config), title="Config", title_align="left")

        if self.summary:
            yield Panel(Pretty(self.summary), title="Summary", title_align="left")

        if self.results:
            yield Panel(Pretty(self.results), title="Results", title_align="left")

        if self.fidelities:
            yield Panel(Pretty(self.fidelities), title="Fidelities", title_align="left")

        if any(self.profiler.profiles):
            yield self.profiler.__rich__()

        if any(self.storage):
            yield Panel(Pretty(self.storage), title="Storage", title_align="left")

        if any(self.plugins):
            yield Panel(Pretty(self.plugins), title="Plugins", title_align="left")

    def __rich__(self) -> RenderableType:
        from rich.console import Group as RichGroup
        from rich.panel import Panel

        from amltk.rich_util import key_with_paren_text

        return Panel(
            RichGroup(*self.rich_renderables()),
            title=key_with_paren_text(
                "Trial",
                self.name,
                key_style="bold",
                val_style="italic",
            ),
            title_align="left",
        )

    class Status(str, Enum):
        """The status of a trial."""

        SUCCESS = "success"
        """The trial was successful."""

        FAIL = "fail"
        """The trial failed."""

        CRASHED = "crashed"
        """The trial crashed."""

        UNKNOWN = "unknown"
        """The status of the trial is unknown."""

        @override
        def __str__(self) -> str:
            return self.value

        def __rich__(self) -> Text:
            from rich.text import Text

            styles = {
                Trial.Status.SUCCESS: "bold green",
                Trial.Status.FAIL: "bold yellow",
                Trial.Status.CRASHED: "bold red",
                Trial.Status.UNKNOWN: "bold underline",
            }

            return Text(self.value, style=styles.get(self, "bold underline"))

    @dataclass
    class Report(Generic[I2]):
        """A report for a trial."""

        trial: Trial[I2]
        """The trial that was run."""

        status: Trial.Status
        """The status of the trial."""

        @property
        def results(self) -> dict[str, Any]:
            """The results of the trial, if any."""
            return self.trial.results

        @property
        def exception(self) -> BaseException | None:
            """The exception of the trial, if any."""
            return self.trial.exception

        @property
        def traceback(self) -> str | None:
            """The traceback of the trial, if any."""
            return self.trial.traceback

        @property
        def name(self) -> str:
            """The name of the trial."""
            return self.trial.name

        @property
        def config(self) -> Mapping[str, Any]:
            """The config of the trial."""
            return self.trial.config

        @property
        def profiles(self) -> Mapping[str, Profile.Interval]:
            """The profiles of the trial."""
            return self.trial.profiles

        @property
        def summary(self) -> dict[str, Any]:
            """The summary of the trial."""
            return self.trial.summary

        @property
        def storage(self) -> set[str]:
            """The storage of the trial."""
            return self.trial.storage

        @property
        def time(self) -> Timer.Interval:
            """The time of the trial."""
            return self.trial.time

        @property
        def memory(self) -> Memory.Interval:
            """The memory of the trial."""
            return self.trial.memory

        @property
        def info(self) -> I2 | None:
            """The info of the trial, specific to the optimizer that issued it."""
            return self.trial.info

        def df(
            self,
            *,
            profiles: bool = True,
            configs: bool = True,
            summary: bool = True,
            results: bool = True,
        ) -> pd.DataFrame:
            """Get a dataframe of the results of the trial.

            !!! note "Prefixes"

                * `summary`: Entries will be prefixed with `#!python "summary:"`
                * `config`: Entries will be prefixed with `#!python "config:"`
                * `results`: Entries will be prefixed with `#!python "results:"`
                * `storage`: Entries will be prefixed with `#!python "storage:"`
                * `profile:<name>`: Entries will be prefixed with
                    `#!python "profile:<name>:"`

            Args:
                profiles: Whether to include the profiles.
                configs: Whether to include the configs.
                summary: Whether to include the summary.
                results: Whether to include the results.
            """
            items = {
                "name": self.name,
                "status": str(self.status),
                "trial_seed": self.trial.seed if self.trial.seed else np.nan,
                "exception": str(self.exception) if self.exception else "NA",
                "traceback": str(self.traceback) if self.traceback else "NA",
            }
            if results:
                items.update(**prefix_keys(self.trial.results, "results:"))
            if summary:
                items.update(**prefix_keys(self.trial.summary, "summary:"))
            if configs:
                items.update(**prefix_keys(self.trial.config, "config:"))
            if profiles:
                for name, profile in sorted(self.profiles.items(), key=lambda x: x[0]):
                    # We log this one seperatly
                    if name == "trial":
                        items.update(profile.to_dict())
                    else:
                        items.update(profile.to_dict(prefix=f"profile:{name}"))

            return pd.DataFrame(items, index=[0]).convert_dtypes().set_index("name")

        @overload
        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: None = None,
        ) -> Any:
            ...

        @overload
        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: type[R],
        ) -> R:
            ...

        def retrieve(
            self,
            key: str,
            *,
            where: str | Path | Bucket[str, Any],
            check: type[R] | None = None,
        ) -> R | Any:
            """Retrieve items related to the trial.

            !!! note "Same argument for `where=`"

                 Use the same argument for `where=` as you did for `store()`.

            ```python exec="true" source="material-block" result="python" title="retrieve" hl_lines="7"
            from amltk.optimization import Trial

            trial = Trial(name="trial", config={"x": 1}, info={})

            trial.store({"config.json": trial.config}, where="./results")
            with trial.begin():
                report = trial.success()

            config = report.retrieve("config.json", where="./results")
            print(config)
            ```

            You could also create a Bucket and use that instead.

            ```python exec="true" source="material-block" result="python" title="retrieve-bucket" hl_lines="11"

            from amltk.optimization import Trial
            from amltk.store import PathBucket

            bucket = PathBucket("results")

            trial = Trial(name="trial", config={"x": 1}, info={})

            trial.store({"config.json": trial.config}, where="./results")

            with trial.begin():
                report = trial.success()

            config = report.retrieve("config.json", where=bucket)
            print(config)
            ```

            Args:
                key: The key of the item to retrieve as said in `.storage`.
                check: If provided, will check that the retrieved item is of the
                    provided type. If not, will raise a `TypeError`. This
                    is only used if `where=` is a `str`, `Path` or `Bucket`.
                where: Where to retrieve the items from.

                    * If a `str` or `Path`, will store
                    a bucket will be created at the path, and the items will be
                    retrieved from a sub-bucket with the name of the trial.

                    * If a `Bucket`, will retrieve the items from a sub-bucket with the
                    name of the trial.

            Returns:
                The retrieved item.

            Raises:
                TypeError: If `check=` is provided and  the retrieved item is not of the provided
                    type.
            """  # noqa: E501
            # If not a Callable, we convert to a path bucket
            method: Bucket[str, Any]
            if isinstance(where, str):
                where = Path(where)
                method = PathBucket(where, create=True)
            elif isinstance(where, Path):
                method = PathBucket(where, create=True)
            else:
                method = where

            # Store in a sub-bucket
            return method.sub(self.name)[key].load(check=check)

        def store(
            self,
            items: Mapping[str, T],
            *,
            where: str | Path | Bucket | Callable[[str, Mapping[str, T]], None],
        ) -> None:
            """Store items related to the trial.

            See: [`Trial.store()`][amltk.optimization.trial.Trial.store]
            """
            self.trial.store(items, where=where)

        @classmethod
        def from_df(cls, df: pd.DataFrame | pd.Series) -> Trial.Report:
            """Create a report from a dataframe.

            See Also:
                * [`.from_dict()`][amltk.optimization.Trial.Report.from_dict]
            """
            if isinstance(df, pd.DataFrame):
                if len(df) != 1:
                    raise ValueError(
                        f"Expected a dataframe with one row, got {len(df)} rows.",
                    )
                series = df.iloc[0]
            else:
                series = df

            data_dict = {"name": series.name, **series.to_dict()}
            return cls.from_dict(data_dict)

        @classmethod
        def from_dict(cls, d: Mapping[str, Any]) -> Trial.Report:
            """Create a report from a dictionary.

            !!! note "Prefixes"

                Please see [`.df()`][amltk.optimization.Trial.Report.df]
                for information on what the prefixes should be for certain fields.

            Args:
                d: The dictionary to create the report from.

            Returns:
                The created report.
            """
            prof_dict = mapping_select(d, "profile:")
            if any(prof_dict):
                profile_names = sorted(
                    {name.rsplit(":", maxsplit=2)[0] for name in prof_dict},
                )
                profiles = {
                    name: Profile.from_dict(mapping_select(prof_dict, f"{name}:"))
                    for name in profile_names
                }
            else:
                profiles = {}

            _trial_profile_items = {
                k: v for k, v in d.items() if k.startswith(("memory:", "time:"))
            }
            if any(_trial_profile_items):
                trial_profile = Profile.from_dict(_trial_profile_items)
                profiles["trial"] = trial_profile
            else:
                trial_profile = Profile.na()

            exception = d.get("exception")
            traceback = d.get("traceback")
            trial_seed = d.get("trial_seed")
            if pd.isna(exception) or exception == "NA":  # type: ignore
                exception = None
            if pd.isna(traceback) or traceback == "NA":  # type: ignore
                traceback = None
            if pd.isna(trial_seed):  # type: ignore
                trial_seed = None

            trial: Trial[None] = Trial(
                name=d["name"],
                config=mapping_select(d, "config:"),
                info=None,  # We don't save this to disk so we load it back as None
                seed=trial_seed,
                fidelities=mapping_select(d, "fidelities:"),
                time=trial_profile.time,
                memory=trial_profile.memory,
                profiler=Profiler(profiles=profiles),
                results=mapping_select(d, "results:"),
                summary=mapping_select(d, "summary:"),
                exception=exception,
                traceback=traceback,
            )
            status = Trial.Status(dict_get_not_none(d, "status", "unknown"))
            if status == Trial.Status.SUCCESS:
                return trial.success(**trial.results)

            if status == Trial.Status.FAIL:
                return trial.fail(**trial.results)

            if status == Trial.Status.CRASHED:
                return trial.crashed(
                    exception=Exception("Unknown status.")
                    if trial.exception is None
                    else None,
                )

            return trial.crashed(exception=Exception("Unknown status."))

        def rich_renderables(self) -> Iterable[RenderableType]:
            """The renderables for rich for this report."""
            from amltk.rich_util import key_val_text

            yield key_val_text("Status", self.status.__rich__())
            yield from self.trial.rich_renderables()

        def __rich__(self) -> Panel:
            from rich.console import Group as RichGroup
            from rich.panel import Panel

            from amltk.rich_util import key_with_paren_text

            return Panel(
                RichGroup(*self.rich_renderables()),
                title=key_with_paren_text(
                    "Trial",
                    self.name,
                    key_style="bold",
                    val_style="italic",
                ),
            )
