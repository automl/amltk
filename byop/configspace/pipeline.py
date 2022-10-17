from __future__ import annotations

from typing import Any

from more_itertools import first_true

from byop.pipeline import Choice, Component, Configurable, Node, Pipeline, choice, step

try:
    from ConfigSpace import Categorical, Configuration, ConfigurationSpace
except ImportError as e:
    raise ImportError(
        "Failed to import ConfigSpace, please install with dependancy"
        " `pip install byop[ConfigSpace]`"
    ) from e


class ConfigSpacePipeline(Pipeline):
    def space(self, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        for node in self:
            if isinstance(node, Configurable):
                cs.add_configuration_space(node.name, node.space, delimiter=":")
            elif isinstance(node, Choice):
                names = [c.name for c in node.choices]
                choice_hp = Categorical(node.name, names, weights=node.weights)
                cs.add_hyperparameter(choice_hp)

                for c in node.choices:
                    choices = self.__class__(c)
                    cs.add_configuration_space(
                        node.name,
                        choices.space(seed),
                        parent_hyperparameter={"parent": choice_hp, "value": c.name},
                    )

        return cs

    def select(
        self, config: Configuration
    ) -> list[tuple[Component | Configurable, dict[str, Any]]]:
        _config = config.get_dictionary()

        def process_choice(
            node: Choice, prefix: str | None = None
        ) -> tuple[Node, dict]:
            substr = prefix + ":" + node.name if prefix is not None else node.name

            # First we try to see if there's a deeper configuration object we need
            # to reach for
            for c in node.choices:
                lsubstr = substr + ":" + c.name
                if any(k.startswith(lsubstr) for k in _config.keys()):
                    return _select(c, substr)

            # We found no deep config entry, it usually means a choice with no
            # configurable component beneath it.
            if substr in _config:
                chosen = _config[substr]
                found = first_true(node.choices, pred=lambda n: n.name == chosen)
                if found is not None:
                    return (found, {})

            raise RuntimeError(
                f"No choice was made for choice `{node.name}` in {_config}"
            )

        def _select(
            node: Node, prefix: str | None = None
        ) -> tuple[Node, dict[str, Any]]:
            if isinstance(node, Choice):
                return process_choice(node, prefix)

            if prefix is None:
                substr = node.name
            else:
                substr = prefix + ":" + node.name
            l = len(substr)

            d = {k[l + 1 :]: v for k, v in _config.items() if k.startswith(substr)}
            return (node, d)

        steps = list(map(_select, self))
        return steps  # type: ignore


if __name__ == "__main__":

    class MyComponent:
        def __init__(self, i: int):
            self.i = i

        def __repr__(self) -> str:
            return f"I am {self.i}"

    space = lambda i: ConfigurationSpace({"i": i})
    _choice = lambda i, *l: choice(f"choice_{i}", *l)
    _step = lambda i: step(str(i), MyComponent, space=space(i))

    pipeline = ConfigSpacePipeline(
        _step(1),
        step("HARDCODED", MyComponent, kwargs={"i": 1337}),
        _choice(
            1.5,
            step("A", MyComponent, kwargs={"i": 1.5}),
            step("B", MyComponent, kwargs={"i": 2.5}),
        ),
        _choice(2, _step(3), _step(4)),
        _step(5),
        _choice(
            6,
            _choice(7, _step(8), _step(9)),
            _choice(10, _step(11), _step(12)),
            _choice(13, _step(14), _step(15)),
        ),
        _choice(
            "test",
            _choice(
                "test_1",
                step("A1", MyComponent, kwargs={"i": 10}),
                step("B1", MyComponent, kwargs={"i": 20}),
            ),
            _choice(
                "test_2",
                step("A2", MyComponent, kwargs={"i": 30}),
                step("B2", MyComponent, kwargs={"i": 40}),
            ),
        ),
    )

    _config = pipeline.space(seed=None).sample_configuration()
    print(pipeline.build(_config))
