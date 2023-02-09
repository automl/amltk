"""Concrete builder for an sklearn pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from result import Err, Ok, Result

from byop.building.builders.builder import Builder, BuilderError
from byop.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline as SklearnPipeline


class SklearnBuilder(Builder["SklearnPipeline"]):
    """Attempts to build a usable object from a pipeline."""

    @classmethod
    def _build(
        cls,
        pipeline: Pipeline,
    ) -> Result[SklearnPipeline, BuilderError | Exception]:
        """Build a pipeline into a usable sklearn pipeline object.

        Args:
            pipeline: The pipeline to build

        Returns:
            Result[SklearnPipeline, Exception]
        """
        try:
            from byop.sklearn.builder import build

            return Ok(build(pipeline))
        except ModuleNotFoundError:
            e = Builder.Error("Could not succesfully import sklearn. Is it installed?")
            return Err(e)
        except Exception as e:  # noqa: BLE001
            return Err(e)
