"""The core step class for the pipeline.

These objects act as a doubly linked list to connect steps into a chain which
are then convenientyl wrapped in a `Pipeline` object. Their concrete implementations
can be found in the `byop.pipeline.components` module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

from attrs import evolve, field, frozen
from more_itertools import consume, last, peekable, triplewise

from byop.typing import Key

if TYPE_CHECKING:
    from byop.pipeline.components import Split

T = TypeVar("T")
SENTINAL = object()

Self = TypeVar("Self")


@frozen(kw_only=True)
class Step(Generic[Key], ABC):
    """The core step class for the pipeline.

    Attributes:
        name: Name of the step
        prv: The previous step in the chain
        nxt: The next step in the chain
    """

    name: Key
    prv: Step[Key] | None = field(default=None, eq=False, repr=False)
    nxt: Step[Key] | None = field(default=None, eq=False, repr=False)

    def __or__(self, nxt: Step[Key]) -> Step[Key]:
        """Append a step on this one, return the head of a new chain of steps.

        Args:
            nxt: The next step in the chain

        Returns:
            Step: The head of the new chain of steps
        """
        if not isinstance(nxt, Step):
            return NotImplemented

        return self.append(nxt)

    def append(self, nxt: Step[Key]) -> Step[Key]:
        """Append a step on this one, return the head of a new chain of steps.

        Args:
            nxt: The next step in the chain

        Returns:
            Step: The head of the new chain of steps
        """
        return Step.join(self, nxt)

    def extend(self, nxt: Iterable[Step[Key]]) -> Step[Key]:
        """Extend many steps on to this one, return the head of a new chain of steps.

        Args:
            nxt: The next steps in the chain

        Returns:
            Step: The head of the new chain of steps
        """
        return Step.join(self, nxt)

    def iter(
        self,
        *,
        backwards: bool = False,
        include_self: bool = True,
        to: Key | Step[Key] | None = None,
    ) -> Iterator[Step[Key]]:
        """Iterate the linked-list of steps.

        Args:
            backwards (optional): Traversal order. Defaults to False
            include_self (optional): Whether to include self in iterator. Default True
            to (optional): Stop iteration at this step. Defaults to None

        Yields:
            Step[Key]: The steps in the chain
        """
        # Break out if current step is `to
        if to is not None:
            if isinstance(to, Step):
                to = to.name
            if self.name == to:
                return

        if include_self:
            yield self

        if backwards:
            if self.prv is not None:
                yield from self.prv.iter(backwards=True, to=to)
        elif self.nxt is not None:
            yield from self.nxt.iter(backwards=False, to=to)

    def head(self) -> Step[Key]:
        """Get the first step of this chain."""
        return last(self.iter(backwards=True))

    def tail(self) -> Step[Key]:
        """Get the last step of this chain."""
        return last(self.iter())

    def proceeding(self) -> Iterator[Step[Key]]:
        """Iterate the steps that follow this one."""
        return self.iter(include_self=False)

    def preceeding(self) -> Iterator[Step[Key]]:
        """Iterate the steps that preceed this one."""
        head = self.head()
        if self != head:
            yield from head.iter(to=self)

    def mutate(self: Self, **kwargs: Any) -> Self:
        """Mutate this step with the given kwargs, will remove any existing nxt or prv.

        Args:
            **kwargs: The attributes to mutate

        Returns:
            Self: The mutated step
        """
        # NOTE: To prevent the confusion that this instance of `step` would link to
        #  `prv` and `nxt` while the steps `prv` and `nxt` would not link to this
        #   mutated step `prv` from the mutated step.
        #   This is unlikely to be very useful for the base Step class other than
        #   to rename it.
        return evolve(self, **{**kwargs, "prv": None, "nxt": None})

    def copy(self: Self) -> Self:
        """Copy this step.

        Returns:
            Self: The copied step
        """
        return copy(self)

    @abstractmethod
    def select(self, choices: Mapping[Key, Key]) -> Iterator[Step[Key]]:
        """Replace the current step with the chosen step if it's a choice.

        Args:
            choices: Mapping of choice names to the path to pick

        Yields:
            Step[Key]: The unmodified step if not a choice, else the chosen choice
                if applicable
        """
        ...

    @abstractmethod
    def remove(self, keys: Sequence[Key]) -> Iterator[Step[Key]]:
        """Remove the given steps from this chain.

        Args:
            keys: The name of the steps to remove

        Yields:
            Step[Key]: The steps in the chain unless it was one to remove
        """
        ...

    @abstractmethod
    def configure(self, configurations: Mapping[Key, Any]) -> Iterator[Step[Key]]:
        """Configure the given steps in this chain.

        Args:
            configurations: The configurations to apply

        Yields:
            Step[Key]: The steps in the chain
        """
        ...

    @abstractmethod
    def walk(
        self,
        splits: Sequence[Split],
        parents: Sequence[Step],
    ) -> Iterator[tuple[list[Split], list[Step], Step]]:
        """Walk along the joined steps, yielding any splits and the parents.

        Args:
            splits: The splits of this step.
            parents: The parents of this step.

        Yields:
            (splits, parents, step):
                Splits to get to this node, direct parents and the current step
        """
        ...

    @abstractmethod
    def replace(self, replacements: Mapping[Key, Step[Key]]) -> Iterator[Step[Key]]:
        """Replace the given step with a new one.

        Args:
            replacements: The steps to replace

        Yields:
            step: The steps in the chain, replaced if in replacements
        """
        ...

    @abstractmethod
    def traverse(self, *, include_self: bool = True) -> Iterator[Step[Key]]:
        """Traverse any sub-steps associated with this step.

        Subclasses should overwrite as required

        Args:
            include_self (optional): Whether to include this step. Defaults to True

        Returns:
            Iterator[Step[Key, O]]: The iterator over steps
        """
        ...

    @classmethod
    def join(cls, *steps: Step[Key] | Iterable[Step[Key]]) -> Step[Key]:
        """Join together a collection of steps, returning the head.

        This is essentially a shortform of Step.chain(*steps) that returns
        the head of the chain. See `Step.chain` for more description.

        Args:
            *steps : Any amount of steps or iterables of steps

        Returns:
            Step[Key]
                The head of the chain of steps
        """
        itr = cls.chain(*steps)
        head = next(itr, None)
        if head is None:
            raise ValueError(f"Recieved no values for {steps=}")

        consume(itr)
        return head

    @classmethod
    def chain(
        cls, *steps: Step[Key] | Iterable[Step[Key]], expand: bool = True
    ) -> Iterator[Step[Key]]:
        """Chain together a collection of steps into an iterable.

        Args:
            *steps : Any amount of steps or iterable of steps.
            expand: Individual steps will be expanded with `step.iter()` while
                Iterables will remain as is, defaults to True

        Returns:
            An iterator over the steps joined together
        """
        expanded = chain.from_iterable(
            (s.iter() if expand else [s]) if isinstance(s, Step) else s for s in steps
        )

        # We use a `peekable` to check if there's actually anything to chain
        # In the off case we got nothing in `*steps` but empty iterables
        new_steps = peekable(copy(s) for s in expanded)
        if not new_steps:
            return

        # As these Steps are frozen, we break the frozen api to build a doubly linked
        # list of steps.
        # ? Is it possible to build a doubly linked list where each node is immutable?
        itr = chain([None], new_steps, [None])
        for a, b, c in triplewise(itr):
            object.__setattr__(b, "prv", a)
            object.__setattr__(b, "nxt", c)
            yield cast("Step[Key]", b)
