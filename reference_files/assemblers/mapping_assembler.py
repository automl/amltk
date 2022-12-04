from __future__ import annotations

from typing import Any, Mapping

from byop.pipeline.step import Choice, Component, Searchable, Split, Step


def _trim(d: Mapping[str, Any]) -> dict[str, Any]:
    d = {k[1:] if k.startswith(":") else k: v for k, v in d.items()}
    return {k: v for k, v in d.items() if k != ""}


def _remove_prefix(s: str, prefix: str) -> str:
    assert s.startswith(prefix), f"{s=} , {prefix=}"
    return s[len(prefix) :]


def mapping_assemble(pipeline: Step, config: Mapping[str, Any]) -> Step:
    """Generate a sequence of Step from the pipeline, selecting from the config

    pipeline : Step
        The pipeline space to assemble from

    config : Mapping[str, Any]
        The config to assemble based upon

    Returns:
        Step: The assembled pipeline steps
    """
    config = _trim(config)
    node = pipeline
    new_node: Step

    if isinstance(node, Searchable):
        updates = _trim(
            {
                _remove_prefix(k, node.name): v
                for k, v in config.items()
                if k.startswith(node.name)
            }
        )
        new_node = node.configure(updates)

    elif isinstance(node, Choice):
        chosen_name = config[node.name]
        chosen_node = node.choose(chosen_name)
        sub_config = {
            _remove_prefix(k, f"{node.name}"): v
            for k, v in config.items()
            if k.startswith(node.name)
        }
        new_node = mapping_assemble(chosen_node, sub_config)

    elif isinstance(node, Split):
        new_node = Split(
            name=node.name,
            paths=[mapping_assemble(step, config) for step in node.paths],
        )

    elif isinstance(node, Component):
        new_node = Component(name=node.name, item=node.o, config=node.config)

    else:
        raise NotImplementedError(node)

    if node.nxt is not None:
        new_nxt = mapping_assemble(node.nxt, config)
        new_node.nxt = new_nxt
        new_nxt.prv = new_node

    return new_node
