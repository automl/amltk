## Metric
A [`Metric`][amltk.optimization.Metric] to let optimizers know how to
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
