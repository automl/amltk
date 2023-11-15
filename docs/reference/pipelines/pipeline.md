## Pieces of a Pipeline
A pipeline is a collection of [`Node`][amltk.pipeline.node.Node]s
that are connected together to form a directed acylic graph, where the nodes
follow a parent-child relation ship. The purpose of these is to form some _abstract_
representation of what you want to search over/optimize and then build into a concrete object.

These [`Node`][amltk.pipeline.node.Node]s allow you to specific the function/object that
will be used there, it's search space and any configuration you want to explicitly apply.
There are various components listed below which gives these nodes extract syntatic meaning,
e.g. a [`Choice`](#choice) which represents some choice between it's children while
a [`Sequential`](#sequential) indicates that each child follows one after the other.

Once a pipeline is created, you can perform 3 very critical operations on it:

* [`search_space(parser=...)`][amltk.pipeline.node.Node.search_space] - This will return the
  search space of the pipeline, as defined by it's nodes. You can find the reference to
  the [available parsers and search spaces here](../pipelines/spaces.md).
* [`configure(config=...)`][amltk.pipeline.node.Node.configure] - This will return a
  new pipeline where each node is configured correctly.
* [`build(builder=...)`][amltk.pipeline.node.Node.build] - This will return some
    concrete object from a configured pipeline. You can find the reference to
    the [available builders here](../pipelines/builders.md).

### Components

::: amltk.pipeline.components
    options:
        members: false

### Node

::: amltk.pipeline.node
    options:
        members: false
