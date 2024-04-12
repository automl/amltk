"""This module contains functionality to construct a pytorch model from a pipeline.

It also includes classes for handling dimension matching between layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from amltk import Choice, Component, Fixed, Node, Sequential
from amltk.exceptions import MatchChosenDimensionsError, MatchDimensionsError


@dataclass
class MatchDimensions:
    """Handles matching dimensions between layers in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the layer name and parameter name
    and stores them for later reference.

    Not intended to be used inside a Choice node.
    """

    layer_name: str
    """The name of the layer."""

    param: str
    """The name of the parameter to match."""

    def evaluate(self, pipeline: Node) -> int:
        """Retrieves the corresponding configuration value from the pipeline.

        Args:
            pipeline: The pipeline to search for the matching configuration.

        Returns:
            The value of the matching configuration parameter.
        """
        layer = pipeline[self.layer_name]
        layer_config = layer.config
        if layer_config is None:
            raise MatchDimensionsError(self.layer_name, None)
        value = layer_config.get(self.param)
        if value is None:
            raise MatchDimensionsError(self.layer_name, self.param)
        return value


@dataclass
class MatchChosenDimensions:
    """Handles matching dimensions for chosen nodes in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the choice name and the corresponding
    dimensions for that choice and stores them for later reference.

    """

    choice_name: str
    """The name of the choice node."""

    choices: Mapping[str, Any]
    """The mapping of choice taken to the dimension to use."""

    def evaluate(self, chosen_nodes: dict[str, str]) -> int:
        """Retrieves the corresponding dimension for the chosen node.

        If the chosen node is not found in the choices dictionary, an error is raised.
        If the dimensions provided are not valid, an error is not raised.
        It is up to the user to ensure that the dimensions are valid.

        Args:
            chosen_nodes: The chosen nodes.

        Returns:
            The value of the matching dimension for a chosen node.
        """
        chosen_node_name = chosen_nodes.get(self.choice_name, None)

        if chosen_node_name is None:
            raise MatchChosenDimensionsError(self.choice_name, chosen_node_name=None)

        try:
            return self.choices[chosen_node_name]
        except KeyError as e:
            raise MatchChosenDimensionsError(self.choice_name, chosen_node_name) from e

    @staticmethod
    def collect_chosen_nodes_names(pipeline: Node) -> dict[str, str]:
        """Collects the names of chosen nodes in the pipeline.

        Each pipeline has a unique set of chosen nodes, which we collect separately
        to handle dimension matching between layers with search spaces.

        Args:
            pipeline: The pipeline containing the model architecture.

        Returns:
            The names of the chosen nodes in the pipeline.
        """
        chosen_nodes_names = {}  # Class variable to store chosen node names

        for node in pipeline.iter():
            if isinstance(node, Choice):
                chosen_node = node.chosen()
                if chosen_node:
                    chosen_nodes_names[node.name] = chosen_node.name

        return chosen_nodes_names


def build_model_from_pipeline(pipeline: Node, /) -> nn.Module:
    """Builds a model from the provided pipeline.

    This function iterates through the pipeline nodes, constructing the model
    layers dynamically based on the node types and configurations. It also
    utilizes the `MatchDimensions` and `MatchChosenDimensions` objects to handle
    dimension matching between layers with search spaces.

    Args:
        pipeline: The pipeline containing the model architecture.

    Returns:
        The constructed PyTorch model.
    """
    model_layers = []

    # Mapping of choice node names to what was chosen for that choice
    chosen_nodes_names = MatchChosenDimensions.collect_chosen_nodes_names(pipeline)

    # NOTE: pipeline.iter() may not be sufficient as we relying on some implied ordering
    # for this to work, i.e. we might not know when we're iterating through nodes of a
    # Join or Split
    for node in pipeline.iter(skip_unchosen=True):
        match node:
            case Fixed(item=built_object):
                model_layers.append(built_object)
            case Component(config=config):
                layer_config = dict(config) if config else {}

                for key, instance in layer_config.items():
                    match instance:
                        case MatchDimensions():
                            layer_config[key] = instance.evaluate(pipeline)
                        case MatchChosenDimensions():
                            layer_config[key] = instance.evaluate(chosen_nodes_names)
                        case _:
                            # Just used the value directly
                            pass

                layer = node.build_item(**layer_config)
                model_layers.append(layer)
            case Sequential() | Choice():
                pass  # Skip these as it will come up in iteration...
            case _:
                # TODO: Support other node types
                raise NotImplementedError(f"Node type {type(node)} not supported yet.")

    return nn.Sequential(*model_layers)
