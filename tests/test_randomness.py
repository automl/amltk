from __future__ import annotations

from typing import TypeVar

import numpy as np
from pytest_cases import case, parametrize_with_cases

from amltk.randomness import as_int, as_randomstate, as_rng
from amltk.types import Seed

S = TypeVar("S", bound=Seed)


@case(tags=["static"])
def case_int_seed() -> tuple[int, int]:
    return 37, 37


@case(tags=["dynamic"])
def case_randomstate_seed() -> tuple[np.random.RandomState, np.random.RandomState]:
    return np.random.RandomState(137), np.random.RandomState(137)


@case(tags=["dynamic"])
def case_generator_seed() -> tuple[np.random.Generator, np.random.Generator]:
    return np.random.default_rng(1337), np.random.default_rng(1337)


@parametrize_with_cases("seeds", cases=".", has_tag="static")
def test_as_int_static(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_int(seed1), as_int(seed2)
    second_i1, second_i2 = as_int(seed1), as_int(seed2)

    # The first time we call it **should** be the same as the second time
    assert first_i1 == second_i1
    assert first_i2 == second_i2


@parametrize_with_cases("seeds", cases=".", has_tag="dynamic")
def test_as_int_dynamic(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_int(seed1), as_int(seed2)
    second_i1, second_i2 = as_int(seed1), as_int(seed2)

    assert first_i1 == first_i2
    assert second_i1 == second_i2

    # The first time we call it **should not** be the same as the second time
    assert first_i1 != second_i1
    assert first_i2 != second_i2


@parametrize_with_cases(argnames="seeds", cases=".", has_tag="static")
def test_as_randomstate_static(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_randomstate(seed1), as_randomstate(seed2)
    second_i1, second_i2 = as_randomstate(seed1), as_randomstate(seed2)

    first_ten_numbers_one = [first_i1.randint(0, 100) for _ in range(10)]
    first_ten_numbers_two = [first_i2.randint(0, 100) for _ in range(10)]

    assert first_ten_numbers_one == first_ten_numbers_two

    second_ten_numbers_one = [second_i1.randint(0, 100) for _ in range(10)]
    second_ten_numbers_two = [second_i2.randint(0, 100) for _ in range(10)]

    assert second_ten_numbers_one == second_ten_numbers_two

    assert first_ten_numbers_one == second_ten_numbers_one
    assert first_ten_numbers_two == second_ten_numbers_two


@parametrize_with_cases(argnames="seeds", cases=".", has_tag="dynamic")
def test_as_randomstate_dynamic(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_randomstate(seed1), as_randomstate(seed2)
    second_i1, second_i2 = as_randomstate(seed1), as_randomstate(seed2)

    first_ten_numbers_one = [first_i1.randint(0, 100) for _ in range(10)]
    first_ten_numbers_two = [first_i2.randint(0, 100) for _ in range(10)]

    assert first_ten_numbers_one == first_ten_numbers_two

    second_ten_numbers_one = [second_i1.randint(0, 100) for _ in range(10)]
    second_ten_numbers_two = [second_i2.randint(0, 100) for _ in range(10)]

    assert second_ten_numbers_one == second_ten_numbers_two

    assert first_ten_numbers_one != second_ten_numbers_one
    assert first_ten_numbers_two != second_ten_numbers_two


@parametrize_with_cases(argnames="seeds", cases=".", has_tag="static")
def test_as_rng_static(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_rng(seed1), as_rng(seed2)
    second_i1, second_i2 = as_rng(seed1), as_rng(seed2)

    first_ten_numbers_one = [first_i1.integers(0, 100) for _ in range(10)]
    first_ten_numbers_two = [first_i2.integers(0, 100) for _ in range(10)]

    assert first_ten_numbers_one == first_ten_numbers_two

    second_ten_numbers_one = [second_i1.integers(0, 100) for _ in range(10)]
    second_ten_numbers_two = [second_i2.integers(0, 100) for _ in range(10)]

    assert second_ten_numbers_one == second_ten_numbers_two

    assert first_ten_numbers_one == second_ten_numbers_one
    assert first_ten_numbers_two == second_ten_numbers_two


@parametrize_with_cases(argnames="seeds", cases=".", has_tag="dynamic")
def test_as_rng_dynamic(seeds: tuple[S, S]):
    seed1, seed2 = seeds

    first_i1, first_i2 = as_rng(seed1), as_rng(seed2)
    second_i1, second_i2 = as_rng(seed1), as_rng(seed2)

    first_ten_numbers_one = [first_i1.integers(0, 100) for _ in range(10)]
    first_ten_numbers_two = [first_i2.integers(0, 100) for _ in range(10)]

    assert first_ten_numbers_one == first_ten_numbers_two

    second_ten_numbers_one = [second_i1.integers(0, 100) for _ in range(10)]
    second_ten_numbers_two = [second_i2.integers(0, 100) for _ in range(10)]

    assert second_ten_numbers_one == second_ten_numbers_two

    assert first_ten_numbers_one != second_ten_numbers_one
    assert first_ten_numbers_two != second_ten_numbers_two
