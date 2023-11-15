## Builders
A [pipeline](../pipelines/pipeline.md) of [`Node`][amltk.pipeline.Node]s
is just an abstract representation of some implementation of a pipeline that will actually do
things, for example an sklearn [`Pipeline`][sklearn.pipeline.Pipeline] or a
Pytorch `Sequential`.

To facilitate custom builders and to allow you to customize building,
there is a explicit argument `builder=` required when
calling [`.build(builder=...)`][amltk.pipeline.Node] on your pipeline.

Each builder gives the [various kinds of components](../pipelines/pipeline.md)
an actual meaning, for example the [`Split`][amltk.pipeline.Split] with
the sklearn [`builder()`][amltk.pipeline.builders.sklearn.build],
translates to a [`ColumnTransformer`][sklearn.compose.ColumnTransformer] and
a [`Sequential`][amltk.pipeline.Sequential] translates to an sklearn
[`Pipeline`][sklearn.pipeline.Pipeline].


## Scikit-learn

::: amltk.pipeline.builders.sklearn
    options:
        members: False

## PyTorch
??? todo "Planned"

    If anyone has good knowledge of building pytorch networks in a more functional
    manner and would like to contribute, please feel free to reach out!

At the moment, we do not provide any native support for `torch`. You can
however make use of `skorch` to convert your networks to a scikit-learn interface,
using the scikit-learn builder instead.
