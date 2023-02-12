"""A module for some useful tools for creating more fluid interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar

from byop.types import Comparable

V = TypeVar("V", bound=Comparable)
P = ParamSpec("P")


@dataclass
class ChainablePredicate(Generic[P]):
    """A predicate that can be chained with other predicates."""

    pred: Callable[P, bool]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        """Evaluate the predicate chain."""
        return self.pred(*args, **kwargs)

    def __and__(self, other: Callable[P, bool]) -> ChainablePredicate[P]:
        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self(*args, **kwargs) and other(*args, **kwargs)

        return ChainablePredicate(_pred)

    def __or__(self, other: Callable[P, bool]) -> ChainablePredicate[P]:
        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self(*args, **kwargs) or other(*args, **kwargs)

        return ChainablePredicate(_pred)


@dataclass
class DelayedOp(Generic[V, P]):
    """A delayed binary operation that can be chained with other operations."""

    left: Callable[P, V]

    def __lt__(self, right: V) -> ChainablePredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) < right

        return ChainablePredicate(op)

    def __le__(self, right: V) -> ChainablePredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) <= right  # type: ignore

        return ChainablePredicate(op)

    def __gt__(self, right: V) -> ChainablePredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) > right

        return ChainablePredicate(op)

    def __ge__(self, right: V) -> ChainablePredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) >= right  # type: ignore

        return ChainablePredicate(op)

    def __eq__(self, right: V) -> ChainablePredicate[P]:  # type: ignore
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) == right

        return ChainablePredicate(op)
