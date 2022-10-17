"""Smac pipelines rely on ConfigSpace"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ConfigSpace import Configuration
from smac import HyperparameterOptimizationFacade
from smac.facade.abstract_facade import AbstractFacade
from smac.scenario import Scenario

from byop.pipeline import Pipeline


class SMACPipeline(ABC):
    def __init__(
        self,
        pipeline: Pipeline,
        optimizer: type[AbstractFacade] = HyperparameterOptimizationFacade,
        scenario_kwargs: dict[str, Any] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if scenario_kwargs is None:
            scenario_kwargs = {}

        scenario = Scenario(**scenario_kwargs)

        self.pipeline = pipeline
        self.optimizer = optimizer(scenario, self.run, **optimizer_kwargs)

    @abstractmethod
    def run(self, config: Configuration, seed: int = 0) -> float:
        ...
