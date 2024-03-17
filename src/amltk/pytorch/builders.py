from __future__ import annotations

from torch import nn, optim

from amltk import Component, Metric, Sequential, Choice, Fixed

from rich import print



def build_model_from_pipeline(pipeline: Sequential) -> nn.Module:
    """Builds a model from the provided pipeline.

    This function iterates through the pipeline nodes, constructing the model
    layers dynamically based on the node types and configurations. It also
    utilizes the `MatchDimensions` objects to handle dimension matching
    between layers with search spaces.

    Args:
        pipeline: The pipeline containing the model architecture.

    Returns:
        The constructed PyTorch model.
    """
    print("Building model from pipeline")
    # print(pipeline)

    model_layers = []

    for node in pipeline.iter():
        print("Node:",  node.name, "isinstance Choice:", isinstance(node, Choice))
        print("-" * 80)

        if isinstance(node, Choice):
            chosen_node = node.chosen()
            print("Choice node")
            print(chosen_node)
            print("*" * 80)

        # Check if node is a Flatten layer, ReLU or similar.
        if isinstance(node, Fixed):
            model_layers.append(node.item)

        # Check if node is a Component with config parameter
        elif isinstance(node.item, type) and issubclass(node.item, nn.Module) and node.config:
            layer_config = node.config or {}

            # Evaluate MatchDimensions objects
            for key, value in layer_config.items():
                if isinstance(value, MatchDimensions):
                    layer_config[key] = value.evaluate(pipeline)

            layer = node.build_item(**layer_config)
            model_layers.append(layer)

    model = nn.Sequential(*model_layers)

    print("-" * 80)
    print("Model built")
    print(model)
    print("-" * 80)

    return model

class MatchDimensions:
    """Class to handle matching dimensions between layers in a pipeline.

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
