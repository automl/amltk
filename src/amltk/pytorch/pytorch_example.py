"""Script for building and evaluating an example of PyTorch MLP model on MNIST dataset.
The script defines functions for constructing a neural network model from a pipeline, training the model,
and evaluating its performance.

References:
- PyTorch MNIST example: https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from amltk import Component, Metric, Sequential, Choice, Fixed

# Change this to optuna if you prefer
# from amltk.optimization.optimizers.optuna import OptunaParser
from amltk.optimization.optimizers.smac import SMACOptimizer

if TYPE_CHECKING:
    from amltk import Node, Trial

from rich import print


class Net(nn.Module):
    """Reference neural network model."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return float(test_loss), float(accuracy)


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
    print(pipeline)

    model_layers = []

    for node in pipeline.iter():

        # Check if node is a Flatten layer, ReLU or similar.
        if isinstance(node, Fixed):
            model_layers.append( node.item)

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


def eval_configuration(
    trial: Trial,
    pipeline: Node,
    device: str = "cpu",  # Change if you have a GPU
    epochs: int = 1,  # Fixed for now
    lr: float = 0.1,  # Fixed for now
    gamma: float = 0.7,  # Fixed for now
    batch_size: int = 64,  # Fixed for now
    log_interval: int = 10,  # Fixed for now
) -> Trial.Report:
    """Evaluates a configuration within the given trial.

    This function trains a model based on the provided pipeline and hyperparameters,
    evaluates its performance, and returns a report containing the evaluation results.

    Args:
        trial: The trial object for storing trial-specific information.
        pipeline: The pipeline defining the model architecture.
        device: The device to use for training and evaluation (default is "cpu").
        epochs: The number of training epochs (default is 1).
        lr: The learning rate for the optimizer (default is 0.1).
        gamma: The gamma value for the learning rate scheduler (default is 0.7).
        batch_size: The batch size for training and evaluation (default is 64).
        log_interval: The interval for logging training progress (default is 10).

    Returns:
        A Trial.Report object containing the evaluation results.
    """
    trial.store({"config.json": pipeline.config})
    torch.manual_seed(trial.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    _device = torch.device(device)
    print("Using device", _device)

    model = (
        pipeline.configure(trial.config)
        .build(builder=build_model_from_pipeline)
        .to(_device)
    )

    with trial.profile("training"):
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data, target = data.to(_device), target.to(_device)

                output = model(data)
                loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        ),
                    )
                    lr_scheduler.step()

    final_train_loss, final_train_acc = test(model, _device, train_loader)
    final_test_loss, final_test_acc = test(model, _device, test_loader)
    trial.summary["final_test_loss"] = final_test_loss
    trial.summary["final_test_accuracy"] = final_test_acc
    trial.summary["final_train_loss"] = final_train_loss
    trial.summary["final_train_accuracy"] = final_train_acc

    return trial.success(accuracy=final_test_acc)


class MatchDimensions:
    """Class to handle matching dimensions between layers in a pipeline.

    This class helps ensure compatibility between layers with search spaces
    during HPO optimization. It takes the layer name and parameter name
    and stores them for later reference.

    When called, it retrieves the corresponding configuration value from
    another layer based on the provided information.
    """

    def __init__(self, layer_name: str, param: str):
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


def main() -> None:
    """Main function to orchestrate the model training and evaluation process.

    This function sets up the training environment, defines the search space
    for hyperparameter optimization, and iteratively evaluates different
    configurations using the SMAC optimizer.

    Returns:
        None
    """
    # Training settings
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.device(_device)

    # Download the dataset
    datasets.MNIST("../data", train=True, download=False)
    datasets.MNIST("../data", train=False, download=False)

    # Define the pipeline with search space for hyperparameter optimization
    pipeline = Sequential(
        nn.Flatten(start_dim=1),
        Component(
            nn.Linear,
            config={
                "in_features": 784,
                "out_features": MatchDimensions("fc2", param="in_features"),
            },
            name="fc1",
        ),
        nn.ReLU(),
        Component(
            nn.Linear,
            space={"in_features": (10, 50), "out_features": (10, 30)},
            name="fc2",
        ),
        Component(
            nn.Linear,
            config={
                "in_features": MatchDimensions("fc2", param="out_features"),
                "out_features": 10,
            },
            name="fc3",
        ),
        Component(nn.LogSoftmax, config={"dim": 1}),
        name="my-mlp-pipeline",
    )

    # Define the metric for optimization
    metric = Metric("accuracy", minimize=False, bounds=(0, 1))

    # Initialize the SMAC optimizer
    optimizer = SMACOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=1,
        bucket="pytorch-experiments",
    )

    # Iteratively evaluate different configurations using the optimizer
    trial = optimizer.ask()
    report = eval_configuration(trial, pipeline, device=_device)
    optimizer.tell(report)
    print(report)


if __name__ == "__main__":
    main()
