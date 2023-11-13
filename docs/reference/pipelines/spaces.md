## Spaces
A common requirement when performing optimization of some pipeline
is to be able to parametrize it. To do so we often think about parametrize
each component separately, with the structure of the pipeline adding additional
constraints.

To facilitate this, we allow the construction of
[piplines](site:reference/pipelines.pipeline.md), where each part
of the pipeline can contains a [`.space`][amltk.pipeline.node.Node.space].
When we wish to extract out the entire search space from the pipeline, we can
call [`search_space(parser=...)`][amltk.pipeline.node.Node.search_space] on the root node
of our pipeline, returning some sort of _space_ object.

Now there are unfortunately quite a few search space implementations out there.
Some support concepts such as forbidden combinations, conditionals and
functional constraints, while others are fully constrained just numerical
parameters. Other reasons to choose a particular space representation is
dependant upon some [`Optimizer`](site:reference/optimization/optimizers.md)
you may wish to use, where typically they will only have one preferred search
space representation.

To generalize over this, AMLTK itself will not care what is in a `.space`
of each part of the pipeline, i.e.

```python exec="true" source="material-block" result="python"
from amltk.pipeline import Component

c = Component(object, space="hmmm, a str space?")
from amltk._doc import doc_print; doc_print(print, c)  # markdown-exec: hide
```

What follow's below is a list of supported parsers you could pass `parser=`
to extract a search space representation.

## ConfigSpace

::: amltk.pipeline.parsers.configspace
    options:
        members: false

## Optuna

::: amltk.pipeline.parsers.optuna
    options:
        members: false
