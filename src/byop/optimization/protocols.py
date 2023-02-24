"""Protocols for the optimization module."""

from __future__ import annotations

from typing import Protocol, TypeVar

R = TypeVar("R", contravariant=True)
Config = TypeVar("Config", covariant=True)


class Optimizer(Protocol[Config, R]):
    """An optimizer protocol.

    An optimizer is an object that can sample a config
    from a space using `ask` and a `tell` to inform
    the optimizer of the result of the sampled config.
    """

    def tell(self, result: R) -> None:
        """Tell the optimizer the result of the sampled config.

        Args:
            result: The result of a sampled config.
        """
        ...

    def ask(self) -> Config:
        """Ask the optimizer for a config to sample.

        Returns:
            A config to sample.
        """
        ...
