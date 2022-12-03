from __future__ import annotations

from itertools import chain

from ConfigSpace import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def parse_uniform(hp: UniformIntegerHyperparameter | UniformFloatHyperparameter) -> str:
    kind = "int" if isinstance(hp, UniformIntegerHyperparameter) else "float"
    s = f"{hp.name} ({kind} in [{hp.lower}, {hp.upper}]) = {hp.default_value}"
    if hp.log:
        s += ", log"
    return s


def parse_normal(hp: NormalFloatHyperparameter | NormalIntegerHyperparameter) -> str:
    kind = "int" if isinstance(hp, NormalIntegerHyperparameter) else "float"
    if hp.lower:
        kindstr = f"{kind} in [{hp.lower}, {hp.upper}]"
    else:
        kindstr = f"{kind}"

    s = f"{hp.name} ({kindstr}): Normal({hp.mu}, {hp.sigma}) = {hp.default_value}"
    if hp.log:
        s += ", log"
    return s


def parse_beta(hp: BetaFloatHyperparameter | BetaIntegerHyperparameter) -> str:
    kind = "int" if isinstance(hp, BetaIntegerHyperparameter) else "float"
    kindstr = f"{kind} in [{hp.lower}, {hp.upper}]"
    s = f"{hp.name} ({kindstr}): Beta({hp.alpha}, {hp.beta}) = {hp.default_value}"
    if hp.log:
        s += ", log"
    return s


def parse_constant(hp: Constant) -> str:
    return f"{hp.name} = {hp.value}"


def parse_categorical(hp: Constant) -> str:
    if hp.weights is not None:
        inner = dict(hp.choices, hp.weights)  # type: ignore
    else:
        inner = hp.choices

    return f"{hp.name} {inner} = {hp.default_value}"


def parse_ordinal(hp: Constant) -> str:
    return f"{hp.name} {hp.sequence} = {hp.default_value}"


_mapping = {
    UniformIntegerHyperparameter: parse_uniform,
    UniformFloatHyperparameter: parse_uniform,
    NormalIntegerHyperparameter: parse_normal,
    NormalFloatHyperparameter: parse_normal,
    BetaIntegerHyperparameter: parse_beta,
    BetaFloatHyperparameter: parse_beta,
    Constant: parse_constant,
    CategoricalHyperparameter: parse_categorical,
    OrdinalHyperparameter: parse_ordinal,
}


def node_repr(space: ConfigurationSpace) -> str:
    lines = [_mapping[hp.__class__](hp) for hp in space.get_hyperparameters()]
    for c in chain(space.get_conditions(), space.get_forbiddens()):
        lines.append(str(c))

    return "\n".join(lines)
