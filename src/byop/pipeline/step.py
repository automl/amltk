"""The core step class for the pipeline.

These objects act as a doubly linked list to connect steps into a chain which
are then convenientyl wrapped in a `Pipeline` object. Their concrete implementations
can be found in the `byop.pipeline.components` module.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from attrs import evolve, field, frozen
from more_itertools import consume, first_true, last, peekable, triplewise

from byop.functional import mapping_select
from byop.types import Config, Seed, Space

if TYPE_CHECKING:
    from typing_extensions import Self

    from byop.pipeline.components import Group
    from byop.pipeline.parser import Parser
    from byop.pipeline.sampler import Sampler

T = TypeVar("T")


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
        * [`group()`][byop.pipeline.api.group]
        * [`split()`][byop.pipeline.api.split]
        * [`searchable()`][byop.pipeline.api.searchable]


    Attributes:
        name: Name of the step
        prv: The previous step in the chain
        nxt: The next step in the chain
        parent: Any [`Group`][byop.pipeline.components.Group] or
            [`Choice`][byop.pipeline.components.Choice] that this step is a part of
            and is the head of the chain.
        config: The configuration for this step
        search_space: The search space for this step
    """

    name: str

    prv: Step | None = field(default=None, eq=False, repr=False)
    nxt: Step | None = field(default=None, eq=False, repr=False)
    parent: Step | None = field(default=None, eq=False, repr=False)

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

    def qualified_name(self, *, delimiter: str = ":") -> str:
        """Get the qualified name of this step.

        This is the name of the step prefixed by the names of all the previous
        groups taken to reach this step in the chain.

        Args:
            delimiter: The delimiter to use between names. Defaults to ":"

        Returns:
            The qualified name
        """
        from byop.pipeline.components import Group

        groups = [s for s in self.climb(include_self=False) if isinstance(s, Group)]
        names = [*reversed([group.name for group in groups]), self.name]
        return delimiter.join(names)

    def configured(self) -> bool:
        """Check if this searchable is configured."""
        return self.search_space is None and self.config is not None

    def configure(self, config: Config, *, prefixed_name: bool | None = None) -> Step:
        """Configure this step and anything following it with the given config.

        Args:
            config: The configuration to apply
            prefixed_name: Whether items in the config are prefixed by the names
                of the steps.
                * If `None`, the default, then `prefixed_name` will be assumed to
                    be `True` if this step has a next step or if the config has
                    keys that begin with this steps name.
                * If `True`, then the config will be searched for items prefixed
                    by the name of the step (and subsequent chained steps).
                * If `False`, then the config will be searched for items without
                    the prefix, i.e. the config keys are exactly those matching
                    this steps search space.

        Returns:
            Step: The configured step
        """
        if prefixed_name is None:
            if any(key.startswith(self.name) for key in config):
                prefixed_name = True
            else:
                prefixed_name = self.nxt is not None

        nxt = (
            self.nxt.configure(config, prefixed_name=prefixed_name)
            if self.nxt
            else None
        )

        this_config: Mapping[str, Any]
        if prefixed_name:
            this_config = mapping_select(config, f"{self.name}:")
        else:
            this_config = deepcopy(config)

        if self.config is not None:
            this_config = {**self.config, **this_config}

        new_self = self.mutate(
            config=this_config if this_config else None,
            search_space=None,
            nxt=nxt,
        )

        if nxt is not None:
            # HACK: This is a hack to to modify the fact `nxt` is a frozen
            # object. Frozen objects do not allow setting attributes after
            # instantiation.
            object.__setattr__(nxt, "prv", new_self)

        return new_self

    def head(self) -> Step:
        """Get the first step of this chain."""
        return last(self.iter(backwards=True))

    def tail(self) -> Step:
        """Get the last step of this chain."""
        return last(self.iter())

    def root(self) -> Step:
        """Climb to the first step of this chain."""
        return last(self.climb())

    def climb(self, *, include_self: bool = True) -> Iterator[Step]:
        """Iterate the steps required to reach the root."""
        if include_self:
            yield self

        if self.prv is not None:
            yield from self.prv.climb()
        elif self.parent is not None:
            yield from self.parent.climb()

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
        #   However this can overwritten by passing "nxt" or "prv" explicitly.
        return evolve(self, **{"prv": None, "nxt": None, **kwargs})  # type: ignore

    def copy(self: Self) -> Self:
        """Copy this step.

        Returns:
            Self: The copied step
        """
        return evolve(self)  # type: ignore

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
        groups: Sequence[Group] | None = None,
        parents: Sequence[Step] | None = None,
    ) -> Iterator[tuple[list[Group], list[Step], Step]]:
        """See `Step.walk`."""
        groups = list(groups) if groups is not None else []
        parents = list(parents) if parents is not None else []
        yield groups, parents, self

        if self.nxt is not None:
            yield from self.nxt.walk(groups=groups, parents=[*parents, self])

    def traverse(
        self,
        *,
        include_self: bool = True,
        backwards: bool = False,
    ) -> Iterator[Step]:
        """Traverse any sub-steps associated with this step.

        Subclasses should overwrite as required

        Args:
            include_self: Whether to include this step. Defaults to True
            backwards: Whether to traverse backwards. This will
                climb linearly until it reaches some head.

        Returns:
            Iterator[Step[Key]]: The iterator over steps
        """
        if include_self:
            yield self

        if backwards:
            if self.prv is not None:
                yield from self.prv.traverse(backwards=True)
            elif self.parent is not None:
                yield from self.parent.traverse(backwards=True)

        if self.nxt is not None:
            yield from self.nxt.traverse()

    @overload
    def find(
        self,
        key: str | Callable[[Step], bool],
        default: T,
        *,
        deep: bool = ...,
    ) -> Step | T:
        ...

    @overload
    def find(
        self,
        key: str | Callable[[Step], bool],
        *,
        deep: bool = ...,
    ) -> Step | None:
        ...

    def find(
        self,
        key: str | Callable[[Step], bool],
        default: T | None = None,
        *,
        deep: bool = True,
    ) -> Step | T | None:
        """Find a step in that's nested deeper from this step.

        Args:
            key: The key to search for or a function that returns True if the step
                is the desired step
            default:
                The value to return if the step is not found. Defaults to None
            deep:
                Whether to search the entire pipeline or just the top layer.

        Returns:
            The step if found, otherwise the default value. Defaults to None
        """
        pred: Callable[[Step], bool]
        pred = key if callable(key) else (lambda step: step.name == key)
        if deep:
            all_steps = chain(self.traverse(), self.traverse(backwards=True))
            return first_true(all_steps, default, pred)  # type: ignore

        all_steps = chain(self.iter(), self.iter(backwards=True))
        return first_true(all_steps, default, pred)  # type: ignore

    def path_to(
        self,
        key: str | Step | Callable[[Step], bool],
        *,
        direction: Literal["forward", "backward"] | None = None,
    ) -> list[Step] | None:
        """Get the path to the given step.

        This includes the path to the step itself.

        ```python exec="true" source="material-block" result="python" title="path_to"
        from byop.pipeline import step, split

        head = (
            step("head", 42)
            | step("middle", 0)
            | split(
                "split",
                step("left", 0),
                step("right", 0),
            )
            | step("tail", 0)
        )

        path = head.path_to("left")
        print([s.name for s in path])

        left = head.find("left")
        path = left.path_to("head")
        print([s.name for s in path])
        ```

        Args:
            key: The step to find
            direction: Specify a particular direction to search in. Defaults to None
                which means search both directions, starting with forwards.

        Returns:
            Iterator[Step[Key]]: The path to the step
        """
        if isinstance(key, Step):
            pred = lambda step: step == key
        elif isinstance(key, str):
            pred = lambda step: step.name == key
        else:
            pred = key

        # We found our target, just return now
        if pred(self):
            return [self]

        if direction in (None, "forward") and self.nxt is not None:  # noqa: SIM102
            if path := self.nxt.path_to(pred, direction="forward"):
                return [self, *path]

        if direction in (None, "backward"):
            back = self.prv or self.parent
            if back and (path := back.path_to(pred, direction="backward")):
                return [self, *path]

            return None

        return None

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
        parser: type[Parser[Space]] | Parser[Space],
        *,
        seed: Seed | None = ...,
    ) -> Space:
        ...

    @overload
    def space(
        self,
        parser: None = None,
        *,
        seed: Seed | None = ...,
    ) -> Any:
        ...

    def space(
        self,
        parser: type[Parser[Space]] | Parser[Space] | None = None,
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
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
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
        duplicates: bool | Iterable[Config] = ...,
        max_attempts: int | None = ...,
    ) -> Config:
        ...

    def sample(
        self,
        space: Space,
        *,
        n: int | None = None,
        sampler: Sampler[Space] | None = None,
        seed: Seed | None = None,
        duplicates: bool | Iterable[Config] = False,
        max_attempts: int | None = 10,
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
            duplicates: If True, allow duplicate samples. If False, make
                sure all samples are unique. If a Iterable, make sure all
                samples are unique and not in the Iterable.
            max_attempts: The number of times to attempt sampling unique
                configurations before giving up. If `None` will keep
                sampling forever until satisfied.

        Returns:
            A configuration sampled from the space of the pipeline
        """
        from byop.pipeline.sampler import Sampler

        return Sampler.try_sample(
            space,
            sampler=sampler,
            n=n,
            seed=seed,
            duplicates=duplicates,
            max_attempts=max_attempts,
        )

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
        new_steps = peekable(s.copy() for s in expanded)
        if not new_steps:
            return

        # Used to check if we have a duplicate name,
        # if so get that step and raise an error
        seen_steps: dict[str, Step] = {}

        # As these Steps are frozen, we break the frozen api to build a doubly linked
        # list of steps.
        # ? Is it possible to build a doubly linked list where each node is immutable?
        itr = chain([None], new_steps, [None])
        for prv, cur, nxt in triplewise(itr):  # pyright: reportGeneralTypeIssues=false
            assert cur is not None

            if cur.name in seen_steps:
                duplicates = (cur, seen_steps[cur.name])
                raise Step.DuplicateNameError(duplicates)

            seen_steps[cur.name] = cur

            object.__setattr__(cur, "prv", prv)
            object.__setattr__(cur, "nxt", nxt)
            yield cast(Step, cur)

    class DelimiterInNameError(ValueError):
        """Raise when a delimiter is found in a name."""

        def __init__(self, step: Step, delimiter: str = ":"):
            """Initialize the exception.

            Args:
                step: The step that contains the delimiter
                delimiter: The delimiter that was found
            """
            self.step = step
            self.delimiter = delimiter

        def __str__(self) -> str:
            delimiter = self.delimiter
            return f"Delimiter ({delimiter=}) in name: {self.step.name} for {self.step}"

    class DuplicateNameError(ValueError):
        """Raise when a duplicate name is found."""

        def __init__(self, steps: tuple[Step, Step]):
            """Initialize the exception.

            Args:
                steps: The steps that have the same name
            """
            self.steps = steps

        def __str__(self) -> str:
            s1, s2 = self.steps
            return f"Duplicate names ({s1.name}) for {s1} and {s2}"

    class ConfigurationError(ValueError):
        """Raise when a configuration is invalid."""

        def __init__(self, step: Step, config: Config, reason: str):
            """Initialize the exception.

            Args:
                step: The step that has the invalid configuration
                config: The invalid configuration
                reason: The reason the configuration is invalid
            """
            self.step = step
            self.config = config
            self.reason = reason

        def __str__(self) -> str:
            return (
                f"Invalid configuration: {self.reason}"
                f" - Given by: {self.step}"
                f" - With config: {self.config}"
            )
