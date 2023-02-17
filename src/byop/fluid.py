"""A module for some useful tools for creating more fluid interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, Protocol, TypeVar

from byop.types import Comparable

V = TypeVar("V", bound=Comparable)
P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class Partial(Protocol[P, R]):
    """A protocol for partial functions."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function."""
        ...


@dataclass
class ChainPredicate(Generic[P]):
    """A predicate that can be chained with other predicates.

    Can be chained with other callables using `&` and `|` operators.

    ```python
    from byop.fluid import ChainPredicate

    def is_even(x: int) -> bool:
        return x % 2 == 0

    def is_odd(x: int) -> bool:
        return x % 2 == 1

    and_combined = ChainPredicate() & is_even & is_odd
    assert and_combined_pred(1) is False

    or_combined = ChainPredicate() & is_even | is_odd
    assert or_combined_pred(1) is True
    ```

    Attributes:
        pred: The predicate to be evaluated.
            Defaults to `None` which defaults to returning `True`
            when called.
    """

    pred: Callable[P, bool] | None = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        """Evaluate the predicate chain."""
        if self.pred is None:
            return True

        return self.pred(*args, **kwargs)

    def __and__(self, other: Callable[P, bool] | None) -> ChainPredicate[P]:
        if other is None:
            return self

        call = other

        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self(*args, **kwargs) and call(*args, **kwargs)

        return ChainPredicate(_pred)

    def __or__(self, other: Callable[P, bool] | None) -> ChainPredicate[P]:
        if other is None:
            return self

        call = other

        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self(*args, **kwargs) or call(*args, **kwargs)

        return ChainPredicate(_pred)

    @classmethod
    def all(cls, *preds: Callable[P, bool]) -> ChainPredicate[P]:
        """Create an all predicate from multiple predicates.

        Args:
            preds: The predicates to combine.

        Returns:
            The combined predicate.
        """

        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return all(pred(*args, **kwargs) for pred in preds)

        return ChainPredicate[P](_pred)

    @classmethod
    def any(cls, *preds: Callable[P, bool]) -> ChainPredicate[P]:
        """Create an any predicate from multiple predicates.

        Args:
            preds: The predicates to combine.

        Returns:
            The combined predicate.
        """

        def _pred(*args: P.args, **kwargs: P.kwargs) -> bool:
            return any(pred(*args, **kwargs) for pred in preds)

        return ChainPredicate[P](_pred)


@dataclass
class DelayedOp(Generic[V, P]):
    """A delayed binary operation that can be chained with other operations.

    Sometimes we want to be able to save a predicate for later evaluation but
    use familiar operators to build it up. This class allows us to do that.

    ```python
    from byop.fluid import DelayedOp
    from dataclasses import dataclass

    @dataclass
    class DynamicThing:
        _x: int

        def value(self) -> int:
            return self._x * 2

    dynamo = DynamicThing(2)

    delayed = DelayedOp(dynamo.value) < 5
    assert delayed() is True

    dynamo._x = 3
    assert delayed() is False
    ```


    Attributes:
        left: The left-hand side of the operation to be evaluated later.
    """

    left: Callable[P, V]

    def __lt__(self, right: V) -> ChainPredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) < right

        return ChainPredicate(op)

    def __le__(self, right: V) -> ChainPredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) <= right  # type: ignore

        return ChainPredicate(op)

    def __gt__(self, right: V) -> ChainPredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) > right

        return ChainPredicate(op)

    def __ge__(self, right: V) -> ChainPredicate[P]:
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) >= right  # type: ignore

        return ChainPredicate(op)

    def __eq__(self, right: V) -> ChainPredicate[P]:  # type: ignore
        def op(*args: P.args, **kwargs: P.kwargs) -> bool:
            return self.left(*args, **kwargs) == right

        return ChainPredicate(op)
