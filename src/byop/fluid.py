"""A module for some useful tools for creating more fluid interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar

from byop.types import Comparable

V = TypeVar("V", bound=Comparable)
P = ParamSpec("P")


@dataclass
class ChainablePredicate(Generic[P]):
    """A predicate that can be chained with other predicates.

    Can be chained with other callables using `&` and `|` operators.

    ```python
    from byop.fluid import ChainablePredicate

    def is_even(x: int) -> bool:
        return x % 2 == 0

    def is_odd(x: int) -> bool:
        return x % 2 == 1

    and_combined = ChainablePredicate(is_even) & is_odd
    assert and_combined_pred(1) is False

    or_combined = ChainablePredicate(is_even) | is_odd
    assert or_combined_pred(1) is True
    ```

    Attributes:
        pred: The predicate to be evaluated.
    """

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
