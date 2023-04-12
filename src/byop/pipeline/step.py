"""The core step class for the pipeline.

These objects act as a doubly linked list to connect steps into a chain which
are then convenientyl wrapped in a `Pipeline` object. Their concrete implementations
can be found in the `byop.pipeline.components` module.
"""
from __future__ import annotations

from copy import copy
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    cast,
    overload,
)

from attrs import evolve, field, frozen
from more_itertools import consume, last, peekable, triplewise

from byop.types import Config, Seed, Space

if TYPE_CHECKING:
    from typing_extensions import Self

    from byop.pipeline.components import Split
    from byop.pipeline.parser import Parser
    from byop.pipeline.sampler import Sampler


@frozen(kw_only=True)
class Step(Generic[Space]):
    """The core step class for the pipeline.

    These are simple objects that are named and linked together to form
    a chain. They are then wrapped in a `Pipeline` object to provide
    a convenient interface for interacting with the chain.

    See Also:
        For creating the concrete implementations of this class, you can use these
        convenience methods.

        * [`step()`][byop.pipeline.api.step]
        * [`choice()`][byop.pipeline.api.choice]
        * [`split()`][byop.pipeline.api.split]
        * [`search()`][byop.pipeline.api.search]


    Attributes:
        name: Name of the step
        prv: The previous step in the chain
        nxt: The next step in the chain
    """

    name: str

    prv: Step | None = field(default=None, eq=False, repr=False)
    nxt: Step | None = field(default=None, eq=False, repr=False)

    config: Mapping[str, Any] | None = field(default=None, hash=False)
    search_space: Space | None = field(default=None, hash=False, repr=False)

    DELIMITER: ClassVar[str] = ":"

    def __or__(self, nxt: Step) -> Step:
        """Append a step on this one, return the head of a new chain of steps.

        Args:
            nxt: The next step in the chain

        Returns:
            Step: The head of the new chain of steps
        """
        if not isinstance(nxt, Step):
            return NotImplemented

        return self.append(nxt)

    def append(self, nxt: Step) -> Step:
        """Append a step on this one, return the head of a new chain of steps.

        Args:
            nxt: The next step in the chain

        Returns:
            Step: The head of the new chain of steps
        """
        return Step.join(self, nxt)

    def extend(self, nxt: Iterable[Step]) -> Step:
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
        to: str | Step | None = None,
    ) -> Iterator[Step]:
        """Iterate the linked-list of steps.

        Args:
            backwards: Traversal order. Defaults to False
            include_self: Whether to include self in iterator. Default True
            to: Stop iteration at this step. Defaults to None

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

    def configured(self) -> bool:
        """Check if this searchable is configured."""
        return self.search_space is None and self.config is not None

    def head(self) -> Step:
        """Get the first step of this chain."""
        return last(self.iter(backwards=True))

    def tail(self) -> Step:
        """Get the last step of this chain."""
        return last(self.iter())

    def proceeding(self) -> Iterator[Step]:
        """Iterate the steps that follow this one."""
        return self.iter(include_self=False)

    def preceeding(self) -> Iterator[Step]:
        """Iterate the steps that preceed this one."""
        head = self.head()
        if self != head:
            yield from head.iter(to=self)

    def mutate(self, **kwargs: Any) -> Self:
        """Mutate this step with the given kwargs, will remove any existing nxt or prv.

        Args:
            **kwargs: The attributes to mutate

        Returns:
            Self: The mutated step
        """
        # NOTE: To prevent the confusion that this instance of `step` would link to
        #  `prv` and `nxt` while the steps `prv` and `nxt` would not link to this
        #   *new* mutated step, we explicitly remove the "prv" and "nxt" attributes
        #   This is unlikely to be very useful for the base Step class other than
        #   to rename it.
        return evolve(self, **{**kwargs, "prv": None, "nxt": None})  # type: ignore

    def copy(self) -> Self:
        """Copy this step.

        Returns:
            Self: The copied step
        """
        return copy(self)

    def remove(self, keys: Sequence[str]) -> Iterator[Step]:
        """Remove the given steps from this chain.

        Args:
            keys: The name of the steps to remove

        Yields:
            Step[Key]: The steps in the chain unless it was one to remove
        """
        if self.name not in keys:
            yield self

        if self.nxt is not None:
            yield from self.nxt.remove(keys)

    def walk(
        self,
        splits: Sequence[Split] | None = None,
        parents: Sequence[Step] | None = None,
    ) -> Iterator[tuple[list[Split], list[Step], Step]]:
        """See `Step.walk`."""
        splits = list(splits) if splits is not None else []
        parents = list(parents) if parents is not None else []
        yield splits, parents, self

        if self.nxt is not None:
            yield from self.nxt.walk(splits, [*parents, self])

    def traverse(self, *, include_self: bool = True) -> Iterator[Step]:
        """Traverse any sub-steps associated with this step.

        Subclasses should overwrite as required

        Args:
            include_self: Whether to include this step. Defaults to True

        Returns:
            Iterator[Step[Key, O]]: The iterator over steps
        """
        if include_self:
            yield self

        if self.nxt is not None:
            yield from self.nxt.traverse()

    def replace(self, replacements: Mapping[str, Step]) -> Iterator[Step]:
        """Replace the given step with a new one.

        Args:
            replacements: The steps to replace

        Yields:
            step: The steps in the chain, replaced if in replacements
        """
        yield replacements.get(self.name, self)

        if self.nxt is not None:
            yield from self.nxt.replace(replacements=replacements)

    def select(self, choices: Mapping[str, str]) -> Iterator[Step]:
        """Replace the current step with the chosen step if it's a choice.

        Args:
            choices: Mapping of choice names to the path to pick

        Yields:
            Step[Key]: The unmodified step if not a choice, else the chosen choice
                if applicable
        """
        yield self

        if self.nxt is not None:
            yield from self.nxt.select(choices)

    @overload
    def space(
        self,
        parser: Parser[Space],
        *,
        seed: Seed | None = None,
    ) -> Space:
        ...

    @overload
    def space(
        self,
        parser: None = None,
        *,
        seed: Seed | None = None,
    ) -> Any:
        ...

    def space(
        self,
        parser: Parser[Space] | None = None,
        *,
        seed: Seed | None = None,
    ) -> Space | Any:
        """Get the search space for this step."""
        from byop.pipeline.parser import Parser

        return Parser.try_parse(self, parser=parser, seed=seed)

    def build(self) -> Any:
        """Build the step.

        Subclasses should overwrite as required, by default for a Step,
        this will raise an Error

        Raises:
            NotImplementedError: If not overwritten
        """
        raise NotImplementedError(f"`build()` is not implemented for {type(self)}")

    @overload
    def sample(
        self,
        space: Space,
        *,
        n: int,
        sampler: Sampler[Space] | None = ...,
        seed: Seed | None = ...,
    ) -> list[Config]:
        ...

    @overload
    def sample(
        self,
        space: Space,
        *,
        n: None = None,
        sampler: Sampler[Space] | None = ...,
        seed: Seed | None = ...,
    ) -> Config:
        ...

    def sample(
        self,
        space: Space,
        *,
        n: int | None = None,
        sampler: Sampler[Space] | None = None,
        seed: Seed | None = None,
    ) -> Config | list[Config]:
        """Sample a configuration from the space of the pipeline.

        Args:
            space: The space to sample from
            n: The number of configurations to sample. If `None`, a single
                configuration will be sampled. If `n` is greater than 1, a list of
                configurations will be returned.
            sampler: The sampler to use. If `None`, a sampler will be automatically
                chosen based on the type of space that is provided.
            seed: The seed to seed the space with if applicable.

        Returns:
            A configuration sampled from the space of the pipeline
        """
        from byop.pipeline.sampler import Sampler

        return Sampler.try_sample(space, sampler=sampler, n=n, seed=seed)

    @classmethod
    def join(cls, *steps: Step | Iterable[Step]) -> Step:
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
        cls,
        *steps: Step | Iterable[Step],
        expand: bool = True,
    ) -> Iterator[Step]:
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
            yield cast(Step, b)
