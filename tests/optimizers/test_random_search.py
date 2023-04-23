from __future__ import annotations

import logging
from typing import Callable

import pytest
from ConfigSpace import Configuration, ConfigurationSpace
from more_itertools import all_unique
from pytest_cases import case, parametrize, parametrize_with_cases

from byop.configspace import ConfigSpaceAdapter
from byop.optimization import RandomSearch
from byop.optuna import OptunaSpaceAdapter
from byop.pipeline import Parser, Sampler, Step, searchable

logger = logging.getLogger(__name__)


@case
def case_int_searchable() -> Step:
    return searchable("my_space", space={"a": (1, 1_000)})


@case
def case_mixed_searchable() -> Step:
    return searchable("my_space", space={"a": (1, 10), "b": (1.0, 10.0)})


def custom_sampler(space: ConfigurationSpace, seed: int) -> Configuration:
    space.seed(seed)
    return space.sample_configuration()


@parametrize(
    "parser, sampler",
    [
        (None, None),
        (OptunaSpaceAdapter, OptunaSpaceAdapter),
        (ConfigSpaceAdapter, ConfigSpaceAdapter),
        (ConfigSpaceAdapter, custom_sampler),
    ],
)
@parametrize_with_cases("step", cases=".", prefix="case_")
def test_random_search_space(
    step: Step,
    parser: type[Parser],
    sampler: type[Sampler] | Callable,
) -> None:
    """Test that the random search space is correct."""
    space = step.space(parser=parser)
    optimizer1 = RandomSearch(space=space, sampler=sampler, seed=42)
    trials1 = [optimizer1.ask() for _ in range(10)]

    assert all_unique(trials1)

    optimizer2 = RandomSearch(space=space, sampler=sampler, seed=42)
    trials2 = [optimizer2.ask() for _ in range(10)]

    assert all_unique(trials2)

    assert trials1 == trials2
    assert (t1.config == t2.config for t1, t2 in zip(trials1, trials2))


def test_random_search_exhausted_with_limited_space() -> None:
    limited_space = searchable(
        "my_space",
        space={"a": ["cat", "dog", "elephant"], "b": ["apple", "honey", "spice"]},
    ).space()

    optimizer = RandomSearch(space=limited_space, seed=42)

    for _ in range(3 * 3):
        optimizer.ask()

    with pytest.raises(RandomSearch.ExhaustedError):
        optimizer.ask()
