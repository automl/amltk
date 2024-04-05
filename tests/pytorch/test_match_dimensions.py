from __future__ import annotations

import pytest
import torch

from amltk import (
    Component,
    Fixed,
    Node,
    Sequential,
)
from amltk.exceptions import MatchDimensionsError
from amltk.pytorch import MatchDimensions, build_model_from_pipeline
from tests.pytorch.common import create_optimizer


class TestMatchDimensions:
    @pytest.fixture(scope="class")
    def valid_pipeline(self) -> Node:
        """Fixture to create a valid pipeline."""
        return Sequential(
            Component(
                torch.nn.Linear,
                config={"in_features": 784},
                space={"out_features": (64, 128)},
                name="fc1",
            ),
            Fixed(torch.nn.Sigmoid(), name="activation"),
            Component(
                torch.nn.Linear,
                config={
                    "in_features": MatchDimensions("fc1", param="out_features"),
                    "out_features": MatchDimensions("fc3", param="in_features"),
                },
                name="fc2",
            ),
            Component(
                torch.nn.Linear,
                space={"in_features": (128, 256)},
                config={"out_features": 10},
                name="fc3",
            ),
            Fixed(torch.nn.LogSoftmax(dim=1), name="log_softmax"),
            name="my-pipeline",
        )

    @pytest.fixture(
        scope="class",
        params=[
            MatchDimensions("non-existing-layer", param="out_features"),
            MatchDimensions("fc1", param="non-existing-param"),
            MatchDimensions("fc1", param=None),
            MatchDimensions(layer_name="", param="out_features"),
        ],
    )
    def invalid_pipeline(self, request) -> Node:
        """Fixture to create several invalid pipelines."""
        return Sequential(
            torch.nn.Flatten(start_dim=1),
            Component(
                torch.nn.Linear,
                config={"in_features": 784},
                space={"out_features": (64, 128)},
                name="fc1",
            ),
            Fixed(torch.nn.Sigmoid(), name="activation"),
            Component(
                torch.nn.Linear,
                config={
                    "in_features": request.param,
                    "out_features": 10,
                },
                name="fc2",
            ),
            Fixed(torch.nn.LogSoftmax(dim=1), name="log_softmax"),
            name="my-pipeline",
        )

    def test_match_dimensions_valid(self, valid_pipeline: Node) -> None:
        """Test for valid pipeline."""
        optimizer = create_optimizer(valid_pipeline)
        trial = optimizer.ask()
        model = valid_pipeline.configure(trial.config).build(
            builder=build_model_from_pipeline,
        )

        assert isinstance(model, torch.nn.Sequential)
        assert len(model) == 5

        assert isinstance(model[0], torch.nn.Linear)
        assert model[0].in_features == 784

        assert isinstance(model[1], torch.nn.Sigmoid)

        assert isinstance(model[2], torch.nn.Linear)
        assert model[2].in_features == model[0].out_features
        assert model[2].out_features == model[3].in_features

        assert isinstance(model[3], torch.nn.Linear)
        assert model[3].out_features == 10

    def test_match_dimensions_invalid(self, invalid_pipeline: Node) -> None:
        """Test for invalid pipeline."""
        optimizer = create_optimizer(invalid_pipeline)
        trial = optimizer.ask()

        with pytest.raises((MatchDimensionsError, KeyError)):
            invalid_pipeline.configure(trial.config).build(
                builder=build_model_from_pipeline,
            )
