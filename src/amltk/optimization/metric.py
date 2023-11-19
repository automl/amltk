"""A [`Metric`][amltk.optimization.Metric] to let optimizers know how to
handle numeric values properly.

A `Metric` is defined by a `.name: str` and whether it is better to `.minimize: bool`
the metric. Further, you can specify `.bounds: tuple[lower, upper]` which can
help optimizers and other code know how to treat metrics.

To easily convert between [`loss`][amltk.optimization.Metric.Value.loss],
[`score`][amltk.optimization.Metric.Value.score] of a
a value in a [`Metric.Value`][amltk.optimization.Metric.Value] object.

If the metric is bounded, you can also get the
[`distance_to_optimal`][amltk.optimization.Metric.Value.distance_to_optimal]
which is the distance to the optimal value.

```python exec="true" source="material-block" result="python"
from amltk.optimization import Metric

acc = Metric("accuracy", minimize=False, bounds=(0.0, 1.0))

acc_value = acc.as_value(0.9)
print(f"Cost: {acc_value.distance_to_optimal}")  # Distance to optimal.
print(f"Loss: {acc_value.loss}")  # Something that can be minimized
print(f"Score: {acc_value.score}")  # Something that can be maximized
```

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Self, override


@dataclass(frozen=True)
class Metric:
    """A metric with a given name, optimal direction, and possible bounds."""

    name: str
    """The name of the metric."""

    minimize: bool = field(kw_only=True, default=True)
    """Whether to minimize or maximize the metric."""

    bounds: tuple[float, float] | None = field(kw_only=True, default=None)
    """The bounds of the metric, if any."""

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lower, upper = self.bounds
            if lower > upper:
                raise ValueError(f"Lower bound {lower} > upper bound {upper}")

            object.__setattr__(self, "bounds", (float(lower), float(upper)))

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
    def worst(self) -> Metric.Value:
        """The worst possible value of the metric."""
        if self.bounds:
            v = self.bounds[1] if self.minimize else self.bounds[0]
            return self.as_value(v)

        v = float("inf") if self.minimize else float("-inf")
        return self.as_value(v)

    @property
    def optimal(self) -> Metric.Value:
        """The optimal value of the metric."""
        if self.bounds:
            v = self.bounds[0] if self.minimize else self.bounds[1]
            return self.as_value(v)
        v = float("-inf") if self.minimize else float("inf")
        return self.as_value(v)

    def as_value(self, value: float | int) -> Metric.Value:
        """Convert a value to an metric value."""
        return Metric.Value(metric=self, value=float(value))

    def __call__(self, value: float | int) -> Metric.Value:
        """Convert a value to an metric value."""
        return Metric.Value(metric=self, value=float(value))

    @dataclass(frozen=True, order=True)
    class Value:
        """A recorded value of an metric."""

        metric: Metric = field(compare=False, hash=True)
        """The metric."""

        value: float = field(compare=True, hash=True)
        """The value of the metric."""

        @property
        def minimize(self) -> bool:
            """Whether to minimize or maximize the metric."""
            return self.metric.minimize

        @property
        def bounds(self) -> tuple[float, float] | None:
            """Whether to minimize or maximize the metric."""
            return self.metric.bounds

        @property
        def name(self) -> str:
            """The name of the metric."""
            return self.metric.name

        @property
        def loss(self) -> float:
            """Convert a value to a loss."""
            if self.minimize:
                return float(self.value)
            return -float(self.value)

        @property
        def score(self) -> float:
            """Convert a value to a score."""
            if self.minimize:
                return -float(self.value)
            return float(self.value)

        @property
        def distance_to_optimal(self) -> float | None:
            """The distance to the optimal value, using the bounds if possible."""
            match self.bounds:
                case None:
                    return None
                case (lower, upper) if lower <= self.value <= upper:
                    if self.minimize:
                        return abs(self.value - lower)
                    return abs(self.value - upper)
                case (lower, upper):
                    raise ValueError(f"Value {self.value} is not within {self.bounds=}")

            return None

        def __float__(self) -> float:
            """Convert a value to a float."""
            return float(self.value)

        @override
        def __eq__(self, __value: object) -> bool:
            """Check if two values are equal."""
            if isinstance(__value, Metric.Value):
                return self.value == __value.value
            if isinstance(__value, float | int):
                return self.value == float(__value)
            return NotImplemented
