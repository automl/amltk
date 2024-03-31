from __future__ import annotations

import pytest
import torch

from amltk import (
    Choice,
    Component,
    Fixed,
    MatchChosenDimensions,
    Sequential,
    build_model_from_pipeline,
)
from amltk.exceptions import MatchChosenDimensionsError
from tests.pytorch.common import create_optimizer


class TestMatchChosenDimensions:
    @pytest.fixture(scope="class")
    def common_pipeline(self) -> callable:
        def _create_pipeline(choices: dict) -> Sequential:
            # Define a pipeline with a Choice class
            return Sequential(
                Choice(
                    Sequential(
                        torch.nn.Linear(in_features=10, out_features=20),
                        name="choice1",
                    ),
                    Sequential(
                        torch.nn.Linear(in_features=5, out_features=10),
                        name="choice2",
                    ),
                    name="my_choice",
                ),
                Component(
                    torch.nn.Linear,
                    config={
                        "in_features": MatchChosenDimensions(
                            choice_name="my_choice",
                            choices=choices,
                        ),
                        "out_features": 30,
                    },
                    name="fc1",
                ),
                Choice(torch.nn.ReLU(), torch.nn.Sigmoid(), name="activation"),
                Component(
                    torch.nn.Linear,
                    config={"in_features": 30, "out_features": 10},
                    name="fc2",
                ),
                Fixed(torch.nn.LogSoftmax(dim=1), name="log_softmax"),
                name="my-pipeline",
            )

        return _create_pipeline

    def test_valid_pipeline(self, common_pipeline: callable) -> None:
        valid_pipeline = common_pipeline(choices={"choice1": 20, "choice2": 10})

        optimizer = create_optimizer(valid_pipeline)
        trial = optimizer.ask()
        model = valid_pipeline.configure(trial.config).build(
            builder=build_model_from_pipeline,
        )

        # Verify that the model is constructed correctly
        assert isinstance(model, torch.nn.Sequential)

        # Conditional check for the second layer's in_features
        if model[0].out_features == 20:
            assert model[1].in_features == 20
        elif model[0].out_features == 10:
            assert model[1].in_features == 10

        assert model[1].out_features == 30

        assert isinstance(model[2], torch.nn.ReLU | torch.nn.Sigmoid)

        assert isinstance(model[4], torch.nn.LogSoftmax)
        assert model[4].dim == 1

    def test_invalid_pipeline(self, common_pipeline: callable) -> None:
        # Modify the common pipeline to create a pipeline with invalid choices
        invalid_pipeline = common_pipeline(choices={"choice123": 123, "choice321": 321})

        optimizer = create_optimizer(invalid_pipeline)
        trial = optimizer.ask()

        with pytest.raises(MatchChosenDimensionsError):
            invalid_pipeline.configure(trial.config).build(
                builder=build_model_from_pipeline,
            )
