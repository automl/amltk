"""Simple HPO loop
# Flags: doc-Runnable

# TODO
"""
from __future__ import annotations

from amltk.pipeline import choice, step
from amltk.smac.optimizer import SMACOptimizer

pipeline = choice(
    "choice",
    step(
        "x",
        object(),
        space={"a": [1, 2, 3]},
        fidelities={"b": (1, 10)},
    ),
    step(
        "y",
        object(),
        space={"a": [1, 2, 3]},
        fidelities={"b": (1.0, 10)},
    ),
    step(
        "z",
        object(),
        space={"a": [1, 2, 3]},
        fidelities={"b": (0.0, 1.0)},
    ),
)

print(pipeline.linearized_fidelity(1))

optimizer = SMACOptimizer.create(
    space=pipeline.space(),
    seed=0,
    fidelities=pipeline.fidelities(),
)

for _i in range(8):
    trial = optimizer.ask()
    assert trial.fidelities is not None
    budget = trial.fidelities["budget"]
    print(budget)

    selected_fidelities = pipeline.linearized_fidelity(budget)
    print(selected_fidelities)

    config = {**trial.config, **selected_fidelities}
    selected_pipeline = pipeline.configure(config)
    print(selected_pipeline)
