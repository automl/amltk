"""The pipeline class used to represent a pipeline of steps

This module exposes a Pipelne class that wraps a chain of `Step`, `Component`,
`Searchable` and `Choice` components, created through the `step`, `choice` and `split`
api functions from `byop.pipeline`.
"""
from __future__ import annotations

from itertools import chain
from typing import Callable, Generic, Iterable, Iterator, Sequence, TypeVar, overload

from attrs import frozen
from more_itertools import duplicates_everseen, first_true

from byop.pipeline.components import Split
from byop.pipeline.step import Key, Step

StepConfig = TypeVar("StepConfig")  # Config for an individual step in a pipeline
T = TypeVar("T")  # Dummy typevar


@frozen(kw_only=True)
class Pipeline(Generic[Key], Sequence[Step[Key]]):
    """Base class implementing search routines over steps."""

    steps: list[Step[Key]]

    @property
    def head(self) -> Step[Key]:
        """The first step in the pipeline"""
        return self.steps[0]

    @property
    def tail(self) -> Step[Key]:
        """The last step in the pipeline"""
        return self.steps[-1]

    @overload
    def __getitem__(self, index: int) -> Step[Key]:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Step[Key]]:
        ...

    def __getitem__(self, index: int | slice) -> Step[Key] | Sequence[Step[Key]]:
        return self.steps.__getitem__(index)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step[Key]]:
        return self.steps.__iter__()

    def iter(self) -> Iterator[Step[Key]]:
        """Iterate over the top layer of the pipeline.

        Yields
        ------
            Step[Key]
        """
        yield from iter(self.steps)

    def traverse(self) -> Iterator[Step[Key]]:
        """Traverse the pipeline in a depth-first manner

        Yields
        ------
            Step[Key]
        """
        yield from chain.from_iterable(step.traverse() for step in self.steps)

    def walk(
        self,
    ) -> Iterator[tuple[list[Split[Key]] | None, list[Step[Key]] | None, Step[Key]]]:
        """Walk the pipeline in a depth-first manner

        This is similar to traverse, but yields the splits that lead to the step along
        with any parents in a chain with that step (which does not include the splits)

        Yields
        ------
            (choices, parents, step)
        """
        yield from self.head.walk()

    @overload
    def find(
        self, key: Key | Callable[[Step[Key]], bool], default: T, *, deep: bool = ...
    ) -> Step[Key] | T:
        ...

    @overload
    def find(
        self, key: Key | Callable[[Step[Key]], bool], *, deep: bool = ...
    ) -> Step[Key] | None:
        ...

    def find(
        self,
        key: Key | Callable[[Step[Key]], bool],
        default: T | None = None,
        *,
        deep: bool = True,
    ) -> Step[Key] | T | None:
        """Find a step in the pipeline

        Args:
            key: The key to search for or a function that returns True if the step
                is the desired step
            default (optional):
                The value to return if the step is not found. Defaults to None
            deep:
                Whether to search the entire pipeline or just the top layer.

        Returns
        -------
            The step if found, otherwise the default value. Defaults to None
        """
        return first_true(
            iterable=self.traverse() if deep else self.iter(),
            pred=key if callable(key) else (lambda step: step.name == key),
            default=default,
        )

    def __or__(self, other: Step[Key] | Pipeline[Key]) -> Pipeline[Key]:
        """Append a step or pipeline to this one and return a new one"""
        return self.append(other)

    def append(self, nxt: Pipeline[Key] | Step[Key]) -> Pipeline[Key]:
        """Append a step or pipeline to this one and return a new one"""
        if isinstance(nxt, Pipeline):
            nxt = nxt.head

        return Pipeline.create(Step.chain(self.steps, nxt.iter()))

    def remove(self, name: Key, *, deep: bool = False) -> Pipeline[Key]:
        """Remove a step from the pipeline

        Args:
            name: The name of the step to remove
            deep: Whether to search the entire pipeline or just the top layer. Defaults
                to False

        Returns
        -------
            A new pipeline with the step removed
        """
        # TODO
        if deep:
            raise NotImplementedError()

        # Mypy seems totally okay with is
        # noinspection PyNoneFunctionAssignment
        old_step = self.find(key=name, deep=deep)
        if old_step is None:
            raise KeyError(f"No step with {name=} in pipeline\n{self}")

        assert isinstance(old_step, Step), "Pycharm was right?"
        return Pipeline.create(old_step.preceeding(), old_step.proceeding())

    def replace(
        self,
        name: Key,
        new_step: Step[Key],
        *,
        deep: bool = False,
    ) -> Pipeline[Key]:
        """Replace a step in the pipeline

        Args:
            name: The name of the step to replace
            new_step: The new step to replace the old one with
            deep: Whether to search the entire pipeline or just the top layer. Defaults
                to False

        Returns
        -------
            A new pipeline with the step replaced
        """
        # TODO
        if deep:
            raise NotImplementedError()

        # Mypy seems totally okay with is
        # noinspection PyNoneFunctionAssignment
        old_step = self.find(key=name, deep=deep)
        if old_step is None:
            raise KeyError(f"No step with key {name} in pipeline\n{self}")

        assert isinstance(old_step, Step), "Pycharm was right?"
        return Pipeline.create(old_step.preceeding(), [new_step], old_step.proceeding())

    def validate(self) -> None:
        """Validate the pipeline for any invariants

        Intended for use as an opt-in during development

        Raises
        ------
            AssertionError
                * If there is a duplicate name of any step in the pipeline
        """
        # Check that we do not have any keys with the same Hash
        dupe_steps = list(duplicates_everseen(self.traverse()))
        assert not any(dupe_steps), f"Duplicates in pipeline {dupe_steps}"

    @classmethod
    def create(
        cls, *steps: Step[Key] | Pipeline[Key] | Iterable[Step[Key]]
    ) -> Pipeline[Key]:
        """Create a pipeline from a sequence of steps"""
        # Expand out any pipelines in the init
        expanded = [s.steps if isinstance(s, Pipeline) else s for s in steps]
        step_sequence = list(Step.chain(*expanded))
        return Pipeline(steps=step_sequence)  # type: ignore
