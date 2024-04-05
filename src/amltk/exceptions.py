"""A module holding a decorator to wrap a function to add a traceback to
any exception raised.
"""
from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar
from typing_extensions import ParamSpec

R = TypeVar("R")
E = TypeVar("E")
P = ParamSpec("P")


def safe_map(
    f: Callable[..., R],
    args: Iterable[Any],
) -> Iterator[R | tuple[Exception, str]]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(arg)
        except Exception as e:  # noqa: BLE001
            yield e, traceback.format_exc()


def safe_starmap(
    f: Callable[..., R],
    args: Iterable[Iterable[Any]],
) -> Iterator[R | tuple[Exception, str]]:
    """Map a function over an iterable, catching any exceptions.

    Args:
        f: The function to map.
        args: The iterable to map over.

    Yields:
        The return value of the function, or the exception raised.
    """
    for arg in args:
        try:
            yield f(*arg)
        except Exception as e:  # noqa: BLE001
            yield e, traceback.format_exc()


class IntegrationNotFoundError(Exception):
    """An exception raised when no integration is found."""

    def __init__(self, name: str) -> None:
        """Initialize the exception.

        Args:
            name: The name of the integration that was not found.
        """
        super().__init__(f"No integration found for {name}.")


class AutomaticParameterWarning(UserWarning):
    """Raised when an "auto" parameter of a function is used
    and triggers some behaviour which would be better explicitly
    set.
    """


class SchedulerNotRunningError(RuntimeError):
    """The scheduler is not running."""


class EventNotKnownError(ValueError):
    """The event is not a known one."""


class NoChoiceMadeError(ValueError):
    """No choice was made."""


class NodeNotFoundError(ValueError):
    """The node was not found."""


class RequestNotMetError(ValueError):
    """Raised when a request is not met."""


class ComponentBuildError(TypeError):
    """Raised when failing to build a component."""


class DuplicateNamesError(ValueError):
    """Raised when duplicate names are found."""


class AutomaticThreadPoolCTLWarning(AutomaticParameterWarning):
    """Raised when automatic threadpoolctl is enabled."""


class ImplicitMetricConversionWarning(UserWarning):
    """A warning raised when a metric is implicitly converted to an sklearn scorer.

    This is raised when a metric is provided with a custom function and is
    implicitly converted to an sklearn scorer. This may fail in some cases
    and it is recommended to explicitly convert the metric to an sklearn
    scorer with `make_scorer` and then pass it to the metric with
    [`Metric(fn=...)`][amltk.optimization.Metric].
    """


class TaskTypeWarning(UserWarning):
    """A warning raised about the task type."""


class AutomaticTaskTypeInferredWarning(TaskTypeWarning, AutomaticParameterWarning):
    """A warning raised when the task type is inferred from the target data."""


class MismatchedTaskTypeWarning(TaskTypeWarning):
    """A warning raised when inferred task type with `task_hint` does not
    match the inferred task type from the target data.
    """


class TrialError(RuntimeError):
    """An exception raised from a trial and it is meant to be raised directly
    to the user.
    """


class CVEarlyStoppedError(RuntimeError):
    """An exception raised when a CV evaluation is early stopped."""


class MatchDimensionsError(KeyError):
    """An exception raised for errors related to matching dimensions in a pipeline."""

    def __init__(self, layer_name: str, param: str | None, *args: Any) -> None:
        """Initialize the exception.

        Args:
            layer_name: The name of the layer.
            param: The parameter causing the error, if any.
            *args: Additional arguments to pass to the exception.
        """
        if param:
            super().__init__(
                f"Error in matching dimensions for layer '{layer_name}'. "
                f"Parameter '{param}' not found in the configuration.",
                *args,
            )
        else:
            super().__init__(
                f"Error in matching dimensions for layer '{layer_name}'."
                f" Configuration not found.",
                *args,
            )


class MatchChosenDimensionsError(KeyError):
    """An exception raised related to matching dimensions for chosen nodes."""

    def __init__(
        self,
        choice_name: str,
        chosen_node_name: str | None = None,
        *args: Any,
    ) -> None:
        """Initialize the exception.

        Args:
            choice_name: The name of the choice that caused the error.
            chosen_node_name: The name of the chosen node if available.
            *args: Additional arguments to pass to the exception.
        """
        if chosen_node_name:
            message = (
                f"Error in matching dimensions for chosen node '{chosen_node_name}' "
                f"of Choice '{choice_name}'. Make sure that the names for "
                f"Choice and MatchChosenDimensions 'choices' parameters match."
            )
        else:
            message = (
                f"Choice name '{choice_name}' is not found in the chosen nodes."
                f"Make sure that the names for Choice and "
                f"MatchChosenDimensions 'choice_name' parameters match."
            )
        super().__init__(message, *args)
