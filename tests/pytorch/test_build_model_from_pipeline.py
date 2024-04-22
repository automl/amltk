from __future__ import annotations

import torch

from amltk import Component, Fixed, Sequential
from amltk.pytorch import build_model_from_pipeline


def test_build_model_from_pipeline():
    # Define a simple pipeline for a multi-layer perceptron
    pipeline = Sequential(
        torch.nn.Flatten(start_dim=1),
        Component(
            torch.nn.Linear,
            config={"in_features": 784, "out_features": 100},
            name="fc1",
        ),
        Fixed(torch.nn.ReLU(), name="activation"),
        Component(
            torch.nn.Linear,
            config={"in_features": 100, "out_features": 10},
            name="fc2",
        ),
        torch.nn.LogSoftmax(dim=1),
        name="my-mlp-pipeline",
    )

    # Build the model from the pipeline
    model = build_model_from_pipeline(pipeline)

    # Verify that the model is constructed correctly
    assert isinstance(model, torch.nn.Sequential)
    assert len(model) == 5  # Check the number of layers in the model

    # Check the layer types and dimensions
    assert isinstance(model[0], torch.nn.Flatten)
    assert isinstance(model[1], torch.nn.Linear)
    assert model[1].in_features == 784
    assert model[1].out_features == 100

    assert isinstance(model[2], torch.nn.ReLU)

    assert isinstance(model[3], torch.nn.Linear)
    assert model[3].in_features == 100
    assert model[3].out_features == 10

    assert isinstance(model[4], torch.nn.LogSoftmax)
    assert model[4].dim == 1
