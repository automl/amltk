"""This module contains functionality to construct a pytorch model from a pipeline.

It also includes classes for handling dimension matching between layers.
"""

from __future__ import annotations

from rich import print
from torch import nn

from amltk import Choice, Fixed, Sequential
from amltk.exceptions import MatchChosenDimensionsError, MatchDimensionsError


class MatchDimensions:
    """Handles matching dimensions between layers in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the layer name and parameter name
    and stores them for later reference.

    Attributes:
        layer_name (str): The name of the layer.
        param (str | None): The name of the parameter.
    """

    def __init__(self, layer_name: str, param: str | None):
        """Initializes the MatchDimensions object.

        Args:
           layer_name (str): The name of the layer.
           param (str | None): The name of the parameter.
        """
        self.layer_name = layer_name
        self.param = param

    def evaluate(self, pipeline: Sequential) -> int:
        """Retrieves the corresponding configuration value from the pipeline.

        Args:
            pipeline: The pipeline to search for the matching configuration.

        Returns:
            The value of the matching configuration parameter.
        """
        layer = pipeline[self.layer_name]
        layer_config = layer.config
        if layer_config is None:
            raise MatchDimensionsError(self.layer_name)
        value = layer_config.get(self.param)
        if value is None:
            raise MatchDimensionsError(self.layer_name, self.param)
        return value


class MatchChosenDimensions:
    """Handles matching dimensions for chosen nodes in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the choice name and the corresponding
    dimensions for that choice and stores them for later reference.

    Attributes:
        choice_name (str): The name of the choice.
        choices (dict): A dictionary containing dimensions for choices.
    """

    def __init__(self, choice_name: str, choices: dict):
        """Initializes the MatchChosenDimensions object.

        Args:
            choice_name (str): The name of the choice.
            choices (dict): A dictionary containing dimensions for choices.
        """
        self.choice_name = choice_name
        self.choices = choices

    def evaluate(self, chosen_nodes) -> int:
        """Retrieves the corresponding dimension for the chosen node.

        Args:
            chosen_nodes: The chosen nodes.

        Returns:
            The value of the matching dimension for a chosen node.
        """
        chosen_node_name = chosen_nodes.get(self.choice_name)

        try:
            return self.choices[chosen_node_name]
        except KeyError:
            raise MatchChosenDimensionsError(self.choice_name, chosen_node_name)

    @staticmethod
    def collect_chosen_nodes_names(pipeline: Sequential) -> dict:
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


def build_model_from_pipeline(pipeline: Sequential) -> nn.Module:
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
    print("Building model for the given pipeline...")
    print(pipeline)

    model_layers = []

    # Collect the names of chosen nodes for the given pipeline
    chosen_nodes_names = MatchChosenDimensions.collect_chosen_nodes_names(pipeline)

    for node in pipeline.iter(skip_unchosen=True):
        # Check if node is a Fixed layer (e.g., Flatten, ReLU)
        if isinstance(node, Fixed):
            model_layers.append(node.item)

        # Check if node is a Component with config parameter
        elif (
            isinstance(node.item, type)
            and issubclass(node.item, nn.Module)
            and node.config
        ):
            layer_config = node.config or {}

            # Evaluate MatchDimensions objects
            for key, instance in layer_config.items():
                if isinstance(instance, MatchDimensions):
                    layer_config[key] = instance.evaluate(pipeline)

                if isinstance(instance, MatchChosenDimensions):
                    layer_config[key] = instance.evaluate(chosen_nodes_names)

            # Build the layer using the updated configuration
            layer = node.build_item(**layer_config)
            model_layers.append(layer)

    model = nn.Sequential(*model_layers)

    print("Model built")
    print(model)

    return model
