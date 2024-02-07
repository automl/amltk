# Check the `main()` function to get started and follow it through.
# Note that performance is irrelevant for now.
# Most of my pytorch stuff as an example is just taken from here.
# https://github.com/pytorch/examples/blob/main/mnist/main.py
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from amltk import Component, Metric, Sequential

# Change this to optuna if you prefer
# from amltk.optimization.optimizers.optuna import OptunaParser
from amltk.optimization.optimizers.smac import SMACOptimizer

if TYPE_CHECKING:
    from amltk import Node, Trial

# This is a nice import :)
from rich import print


# NOTE: This is the reference model, slowly try to build up to this
# but make it parametrizable.
# The goal would be that users don't define this class (maybe?)
# but they can define it using the pipeline structure.
# We can handle that later, for now, the pipeline definition below should be enough.
class Net(nn.Module):
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


# Just taken from the pytorch example
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


# NOTE: The idea for this would be to integrate a general enough builder
# into AMLTK that can take a pipeline and build a nn.Module out of it.
def some_custom_building_function(pipeline: Node) -> nn.Module:
    # TODO: This somehow has to go from a configured pipeline to a nn.Module
    # Take a look at the amltk.pipeline.builders.sklearn to see how this is done
    # for sklearn.
    #
    # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    print("Building model from pipeline")
    print(pipeline)

    # TODO: The main difficulty here will be to figure out how to build
    # this correctly given the pipeline configs and the `item` in the pipeline.
    # This means you should manually place in things like `nn.Flatten`,
    # they're already defined in the `main()` function below.
    # Specifically matching input and output dimensions properly, without
    # knowledge ahead of time what the pipeline should be

    model_layers = []
    input_features = None

    # Traverse the pipeline and construct model layers dynamically
    for node in pipeline.iter():
        if isinstance(node.item, type) and issubclass(node.item, nn.Module):
            # Handle components that represent PyTorch layers
            layer_config = node.config or {}
            layer = node.item(**layer_config)
            model_layers.append(layer)

            if isinstance(layer, nn.Linear):
                input_features = layer_config.get("in_features", None)

    # Assemble the model
    if input_features is not None:
        model_layers.insert(0, nn.Flatten(start_dim=1))
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
    trial.store({"config.json": pipeline.config})
    # TODO: I don't know if this is good enough for seeding and if it works across processes
    # for torch?? At least with sklearn you can pass around a RandomState but torch has no
    # such thing
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
        pipeline
        .configure(trial.config)
        .build(builder=some_custom_building_function)  # TODO: This part is where difficulty lies
        .to(_device)
    )

    with trial.profile("training"):
        # I feel like the optimizer and lr_scheduler should somehow also
        # be part of the pipeline that's gotten when calling build
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        # Just a defactor torch training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data, target = data.to(_device), target.to(_device)

                output = model(data)
                loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    # Might want to store these things in the summary, see below
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item()))
                    lr_scheduler.step()

    # if trial.exception:
    #     return trial.fail()

    final_train_loss, final_train_acc = test(model, _device, train_loader)
    final_test_loss, final_test_acc = test(model, _device, test_loader)
    trial.summary["final_test_loss"] = final_test_loss
    trial.summary["final_test_accuracy"] = final_test_acc
    trial.summary["final_train_loss"] = final_train_loss
    trial.summary["final_train_accuracy"] = final_train_acc

    # TODO: We might also want to be able do this inside the training loop,
    # during the batch_idx % log_interval == 0 block.
    # However we would then have to store it as
    #
    #   trial.summary["epoch_{epoch}:batch_{batch_idx}:loss"] = batch_loss
    #   trial.summary["epoch_{epoch}:batch_{batch_idx}:acc"] = batch_accuracy
    #
    # This is not ideal because getting a curve out of this wouldn't work well.
    # It could be possible to do
    #
    # At start,
    #
    #   trial.summary["blahhhh"] = {"loss": [], "acc": []}
    #
    # and then during the loop
    #
    #   trial.summary["blahhhh"]["loss"].append(batch_loss)
    #   trial.summary["blahhhh"]["acc"].append(batch_acc)

    # We need a custom PathLoader to now how to store
    # a .pt file?
    # trial.store({"model.pt": model.state_dict()})

    # Ideally we should have a validation set for doing proper HPO
    # setup but we'll just use the test accuracy
    return trial.success(accuracy=final_test_acc)


def main() -> None:
    # Training settings
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.device(_device)

    # Download the dataset
    datasets.MNIST("../data", train=True, download=False)
    datasets.MNIST("../data", train=False, download=False)

    # TODO: The goal here will be to somehow setup a search space where
    # we can search over the this `20` number, lets say from `10` to `30`??
    # If you find this impossible to do, please write up how you'd like to express it instead
    # and we will go from there.

    pipeline = Sequential(
        nn.Flatten(start_dim=2),  # <- Will be a `Fixed` because it's an instantiated object
        Component(nn.Linear, config={"in_features": 784, "out_features": 20}, name="fc1"),
        # Instead of config pass space
        nn.ReLU,  # <- Will be a `Component` because it's a class
        # Repeat(nn.Conv2d, config={"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1}, name="conv1"), min=1, max=3),
        Component(nn.Linear, config={"in_features": 20, "out_features": 10}, name="fc2"),
        Component(nn.LogSoftmax, config={"dim": 1}),
        name="my-mlp-pipeline",
    )
    # NOTE: I don't particularly like that you have to wrap F.relu in a `Fixed`.
    # * Fixed - Something that is Fixed and doesn't need to be initialized
    # * Component - Something that needs to be initialized with a config
    #
    # The problem is that right now, if we detect a function, we assume it constructs
    # something to use in a pipeline, not that we should use the function directly.

    # FYI: The `Metric` class is, so you don't have to worry about giving
    # the correct thing to the optimizer, the Metric class takes care of
    # normalizing and return a number the optimizer should optimize.
    # Some optimizers always minimize, some can allow you to choose, some
    # work better with normalized values, etc.
    metric = Metric("accuracy", minimize=False, bounds=(0, 1))
    optimizer = SMACOptimizer.create(
        space=pipeline,
        metrics=metric,
        seed=1,
        bucket="pytorch-experiments",
    )

    # We won't use the Scheduler here as it's not needed for making
    # this example work. We'll just use one trial for now.

    # Explanation: run this in the for loop, to get new configurations
    trial = optimizer.ask()
    report = eval_configuration(trial, pipeline, device=_device)
    optimizer.tell(report)
    print(report)


if __name__ == "__main__":
    main()
