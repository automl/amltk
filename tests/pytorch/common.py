from __future__ import annotations

from amltk import Metric, Node
from amltk.optimization.optimizers.smac import SMACOptimizer


def create_optimizer(pipeline: Node) -> SMACOptimizer:
    """Create optimizer for the given pipeline."""
    metric = Metric("accuracy", minimize=False, bounds=(0, 1))
    return SMACOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=1,
        bucket="pytorch-experiments",
    )
