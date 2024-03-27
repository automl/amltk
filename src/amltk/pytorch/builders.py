from __future__ import annotations

from rich import print
from torch import nn

from amltk import Choice, Fixed, Sequential


class MatchDimensions:
    """Handles matching dimensions between layers in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the layer name and parameter name
    and stores them for later reference.

    When called, it retrieves the corresponding configuration value from
    another layer based on the provided information.
    """

    def __init__(self, layer_name: str, param: str | None):
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
            raise ValueError(f"Configuration not found for layer {self.layer_name}")
        value = layer_config.get(self.param)
        if value is None:
            raise ValueError(
                f"Parameter {self.param} not found in config of layer {self.layer_name}",
            )
        return value


class MatchChosenDimensions:
    """Handles matching dimensions for chosen nodes (children of Choice class) in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the choice name and the corresponding
    dimensions for that choice and stores them for later reference.

    When called, it retrieves the corresponding dimension based on the name
    of the chosen node.
    """

    chosen_nodes = {}  # Class variable to store chosen node names

    def __init__(self, choice_name: str, choices: dict):
        self.choice_name = choice_name
        self.choices = choices

    @staticmethod
    def collect_chosen_nodes(pipeline: Sequential) -> None:
        """Collects the names of chosen nodes in the pipeline.

        Args:
            pipeline: The pipeline containing the model architecture.
        """
        for node in pipeline.iter():
            if isinstance(node, Choice):
                chosen_node = node.chosen()
                if chosen_node:
                    MatchChosenDimensions.chosen_nodes[node.name] = chosen_node.name

    def evaluate(self) -> int:
        """Retrieves the corresponding dimension for the chosen node.

        Returns:
            The value of the matching dimension for a chosen node.
        """
        chosen_node_name = MatchChosenDimensions.chosen_nodes[self.choice_name]

        try:
            return self.choices[chosen_node_name]
        except KeyError:
            raise ValueError(
                "Failed to find matching dimension for the chosen node. "
                "Ensure matching names are provided for the Choice.",
            )


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
    model_layers = []

    MatchChosenDimensions.collect_chosen_nodes(pipeline)

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
                    layer_config[key] = instance.evaluate()

            # Build the layer using the updated configuration
            layer = node.build_item(**layer_config)
            model_layers.append(layer)

    model = nn.Sequential(*model_layers)

    print("-" * 80)
    print("Model built")
    print(model)
    print("-" * 80)

    return model
