"""A [`Metric`][amltk.optimization.Metric] to let optimizers know how to
handle numeric values properly.

A `Metric` is defined by a `.name: str` and whether it is better to `.minimize: bool`
the metric. Further, you can specify `.bounds: tuple[lower, upper]` which can
help optimizers and other code know how to treat metrics.

To easily convert between `loss` and
`score` of some value you can use the [`loss()`][amltk.optimization.Metric.loss]
and [`score()`][amltk.optimization.Metric.score] methods.

If the metric is bounded, you can also make use of the
[`distance_to_optimal()`][amltk.optimization.Metric.distance_to_optimal]
function which is the distance to the optimal value.

In the case of optimization, we provide a
[`normalized_loss()`][amltk.optimization.Metric.normalized_loss] which
normalized the value to be a minimization loss, that is also bounded
if the metric itself is bounded.

```python exec="true" source="material-block" result="python"
from amltk.optimization import Metric

acc = Metric("accuracy", minimize=False, bounds=(0, 100))

print(f"Distance: {acc.distance_to_optimal(90)}")  # Distance to optimal.
print(f"Loss: {acc.loss(90)}")  # Something that can be minimized
print(f"Score: {acc.score(90)}")  # Something that can be maximized
print(f"Normalized loss: {acc.normalized_loss(90)}")  # Normalized loss
```

"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec
from typing_extensions import Self, override

import numpy as np

if TYPE_CHECKING:
    from sklearn.metrics._scorer import _MultimetricScorer, _Scorer


P = ParamSpec("P")

SklearnResponseMethods = Literal["predict", "predict_proba", "decision_function"]


@dataclass(frozen=True)
class Metric(Generic[P]):
    """A metric with a given name, optimal direction, and possible bounds."""

    name: str
    """The name of the metric."""

    minimize: bool = field(kw_only=True, default=True)
    """Whether to minimize or maximize the metric."""

    bounds: tuple[float, float] | None = field(kw_only=True, default=None)
    """The bounds of the metric, if any."""

    fn: Callable[P, float] | None = field(kw_only=True, default=None, compare=False)
    """A function to attach to this metric to be used within a trial."""

    class Comparison(str, Enum):
        """The comparison between two values."""

        BETTER = "better"
        WORSE = "worse"
        EQUAL = "equal"

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lower, upper = self.bounds
            if lower > upper:
                raise ValueError(f"Lower bound {lower} > upper bound {upper}")

            object.__setattr__(self, "bounds", (float(lower), float(upper)))

        if self.name[0].isdigit():
            raise ValueError(
                f"Metric name {self.name} cannot start with a digit."
                " Must be a valid Python identifier.",
            )

        for c in "[](){}<>|&^%$#@!~`":
            if c in self.name:
                raise ValueError(
                    f"Metric name {self.name} cannot contain '{c}'."
                    " Must be a valid Python identifier.",
                )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> float:
        """Call the associated function with this metric."""
        if self.fn is None:
            raise ValueError(
                f"Metric {self.name} does not have a function to call."
                " Please provide a function to `Metric(fn=...)` if you"
                " want to call this metric like this.",
            )
        return self.fn(*args, **kwargs)

    def as_scorer(
        self,
        *,
        response_method: (
            SklearnResponseMethods | Sequence[SklearnResponseMethods] | None
        ) = None,
        **scorer_kwargs: Any,
    ) -> _Scorer:
        """Convert a metric to a sklearn scorer.

        Args:
            response_method: The response method to use for the scorer.
                This can be a single method or an iterable of methods.
            scorer_kwargs: Additional keyword arguments to pass to the
                scorer during the call. Forwards to [`sklearn.metrics.make_scorer`][].

        Returns:
            The sklearn scorer.
        """
        from sklearn.metrics import get_scorer, make_scorer

        match self.fn:
            case None:
                try:
                    return get_scorer(self.name)
                except ValueError as e:
                    raise ValueError(
                        f"Could not find scorer for {self.name}."
                        " Please provide a function to `Metric(fn=...)`.",
                    ) from e
            case fn:
                return make_scorer(
                    fn,
                    greater_is_better=not self.minimize,
                    response_method=response_method,
                    **scorer_kwargs,
                )

    @override
    def __str__(self) -> str:
        parts = [self.name]
        if self.bounds is not None:
            parts.append(f"[{self.bounds[0]}, {self.bounds[1]}]")
        parts.append(f"({'minimize' if self.minimize else 'maximize'})")

        return " ".join(parts)

    @classmethod
    def from_str(cls, s: str) -> Self:
        """Create an metric from a str.

        ```python exec="true" source="material-block" result="python"
        from amltk.optimization import Metric

        s = "loss (minimize)"
        metric = Metric.from_str(s)
        print(metric)

        s = "accuracy [0.0, 1.0] (maximize)"
        metric = Metric.from_str(s)
        print(metric)
        ```

        Args:
            s: The string to parse.

        Returns:
            The parsed metric.
        """
        splits = s.split(" ")
        # No bounds
        if len(splits) == 2:  # noqa: PLR2004
            name, minimize_str = splits
            bounds = None
        else:
            name, lower_str, upper_str, minimize_str = splits
            bounds = (float(lower_str[1:-1]), float(upper_str[:-1]))

        minimize = minimize_str == "(minimize)"
        return cls(name=name, minimize=minimize, bounds=bounds)

    @property
    def worst(self) -> float:
        """The worst possible value of the metric."""
        if self.bounds is not None:
            return self.bounds[1] if self.minimize else self.bounds[0]

        return float("inf") if self.minimize else float("-inf")

    @property
    def optimal(self) -> float:
        """The optimal value of the metric."""
        if self.bounds:
            return self.bounds[0] if self.minimize else self.bounds[1]

        return float("-inf") if self.minimize else float("inf")

    def distance_to_optimal(self, v: float) -> float:
        """The distance to the optimal value, using the bounds if possible."""
        match self.bounds:
            case None:
                raise ValueError(
                    f"Metric {self.name} is unbounded, can not compute distance"
                    " to optimal.",
                )
            case (lower, upper) if lower <= v <= upper:
                if self.minimize:
                    return abs(v - lower)
                return abs(v - upper)
            case (lower, upper):
                raise ValueError(f"Value {v} is not within {self.bounds=}")
            case _:
                raise ValueError(f"Invalid {self.bounds=}")

    def normalized_loss(self, v: float) -> float:
        """The normalized loss of a value if possible.

        If both sides of the bounds are finite, we can normalize the value
        to be between 0 and 1.
        """
        match self.bounds:
            # If both sides are finite, we can 0-1 normalize
            case (lower, upper) if not np.isinf(lower) and not np.isinf(upper):
                cost = (v - lower) / (upper - lower)
                cost = 1 - cost if self.minimize is False else cost
            # No bounds or one unbounded bound, we can't normalize
            case _:
                cost = v if self.minimize else -v

        return cost

    def loss(self, v: float, /) -> float:
        """Convert a value to a loss."""
        return float(v) if self.minimize else -float(v)

    def score(self, v: float, /) -> float:
        """Convert a value to a score."""
        return -float(v) if self.minimize else float(v)

    def compare(self, v1: float, v2: float) -> Metric.Comparison:
        """Check if `v1` is better than `v2`."""
        minimize = self.minimize
        if v1 == v2:
            return Metric.Comparison.EQUAL
        if v1 > v2:
            return Metric.Comparison.WORSE if minimize else Metric.Comparison.BETTER

        # v1 < v2
        return Metric.Comparison.BETTER if minimize else Metric.Comparison.WORSE


@dataclass(frozen=True, kw_only=True)
class MetricCollection(Mapping[str, Metric]):
    """A collection of metrics."""

    metrics: Mapping[str, Metric] = field(default_factory=dict)
    """The metrics in this collection."""

    @override
    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    @override
    def __len__(self) -> int:
        return len(self.metrics)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.metrics)

    def as_sklearn_scorer(
        self,
        *,
        response_methods: (
            Mapping[str, SklearnResponseMethods | Sequence[SklearnResponseMethods]]
            | None
        ) = None,
        scorer_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        raise_exc: bool = True,
    ) -> _MultimetricScorer:
        """Convert this collection to a sklearn scorer."""
        from sklearn.metrics._scorer import _MultimetricScorer

        rms = response_methods or {}
        skwargs = scorer_kwargs or {}

        scorers = {
            k: v.as_scorer(response_method=rms.get(k), **skwargs.get(k, {}))
            for k, v in self.items()
        }
        return _MultimetricScorer(scorers=scorers, raise_exc=raise_exc)

    def optimums(self) -> Mapping[str, float]:
        """The optimums of the metrics."""
        return {k: v.optimal for k, v in self.items()}

    def worsts(self) -> Mapping[str, float]:
        """The worsts of the metrics."""
        return {k: v.worst for k, v in self.items()}

    @classmethod
    def from_empty(cls) -> MetricCollection:
        """Create an empty metric collection."""
        return cls(metrics={})

    @classmethod
    def from_collection(
        cls,
        metrics: Metric | Iterable[Metric] | Mapping[str, Metric],
    ) -> MetricCollection:
        """Create a metric collection from an iterable of metrics."""
        match metrics:
            case Metric():
                return cls(metrics={metrics.name: metrics})
            case Mapping():
                return MetricCollection(metrics={m.name: m for m in metrics.values()})
            case Iterable():
                return cls(metrics={m.name: m for m in metrics})  # type: ignore
            case _:
                raise TypeError(
                    f"Expected a Metric, Iterable[Metric], or Mapping[str, Metric]."
                    f" Got {type(metrics)} instead.",
                )
