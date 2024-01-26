from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import threadpoolctl

from amltk import Component, Metric, Node, Trial
from amltk.optimization.history import History
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.scheduling.scheduler import Scheduler
from amltk.scheduling.task import Task
from amltk.store import PathBucket
from amltk.types import Seed

METRIC = Metric("acc", minimize=False, bounds=(0.0, 1.0))


class _CustomError(Exception):
    pass


def target_funtion(trial: Trial, pipeline: Node) -> Trial.Report:  # noqa: ARG001
    # We don't really care here
    with trial.begin():
        pass

    threadpool_info = threadpoolctl.threadpool_info()
    trial.summary["num_threads"] = threadpool_info[0]["num_threads"]
    return trial.success(acc=0.5)


def test_custom_callback_used(tmp_path: Path) -> None:
    def my_callback(task: Task, scheduler: Scheduler, history: History) -> None:  # noqa: ARG001
        raise _CustomError()

    component = Component(object, space={"a": (0.0, 1.0)})

    with pytest.raises(_CustomError):
        component.optimize(
            target_funtion,
            metric=METRIC,
            on_begin=my_callback,
            max_trials=1,
            working_dir=tmp_path,
        )


def test_populates_given_history(tmp_path: Path) -> None:
    history = History()
    component = Component(object, space={"a": (0.0, 1.0)})
    trial = Trial(name="test_trial", config={})
    with trial.begin():
        pass
    report = trial.success()
    history.add(report)

    component.optimize(
        target_funtion,
        metric=METRIC,
        history=history,
        max_trials=1,
        working_dir=tmp_path,
    )


def test_custom_create_optimizer_signature(tmp_path: Path) -> None:
    component = Component(object, space={"a": (0.0, 1.0)})

    def my_custom_optimizer_creator(
        *,
        space: Node,
        metrics: Metric | Sequence[Metric],
        bucket: PathBucket | None = None,
        seed: Seed | None = None,
    ) -> SMACOptimizer:
        assert space is component
        assert metrics is METRIC
        assert bucket is not None
        assert bucket.path == tmp_path
        assert seed == 1

        raise _CustomError()

    with pytest.raises(_CustomError):
        component.optimize(
            target_funtion,
            metric=METRIC,
            optimizer=my_custom_optimizer_creator,
            max_trials=1,
            seed=1,
            working_dir=tmp_path,
        )


def test_history_populated_with_exactly_maximum_trials(tmp_path: Path) -> None:
    component = Component(object, space={"a": (0.0, 1.0)})
    history = component.optimize(
        target_funtion,
        metric=METRIC,
        max_trials=10,
        working_dir=tmp_path,
    )
    assert len(history) == 10


def test_sklearn_head_triggers_triggers_threadpoolctl(tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    info = threadpoolctl.threadpool_info()
    num_threads = info[0]["num_threads"]

    component = Component(RandomForestClassifier, space={"a": (0.0, 1.0)})
    history = component.optimize(
        target_funtion,
        metric=METRIC,
        max_trials=1,
        working_dir=tmp_path,
    )

    report = history[0]
    # Should have a different number of threads in there. By default 1
    assert report.summary["num_threads"] != num_threads
    assert report.summary["num_threads"] == 1


def test_no_sklearn_head_does_not_trigger_threadpoolctl(tmp_path: Path) -> None:
    info = threadpoolctl.threadpool_info()
    num_threads = info[0]["num_threads"]

    component = Component(object, space={"a": (0.0, 1.0)})
    history = component.optimize(
        target_funtion,
        metric=METRIC,
        max_trials=1,
        working_dir=tmp_path,
    )

    report = history[0]
    assert report.summary["num_threads"] == num_threads
