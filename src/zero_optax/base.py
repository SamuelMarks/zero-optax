"""Base classes, states, and protocols for zero-optax gradient transformations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Protocol, Union

from zero_optax._tree import tree_zeros_like
from zero_optax._types import (
    Array,
    NamedTupleProtocol,
    Params,
    PyTree,
    ScalarOrSchedule,
    Schedule,
    Tensor,
    Updates,
)


class EmptyState(NamedTuple):
    """State for stateless gradient transformations."""


class TraceState(NamedTuple):
    """State for momentum/trace accumulation transformations.

    Attributes:
        trace: Rolling momentum trace PyTree matching parameters structure.
    """

    trace: Updates


class ScaleByScheduleState(NamedTuple):
    """State holding the integer step count for schedule-based scaling.

    Attributes:
        count: Current optimization step count Tensor.
    """

    count: Tensor


class ScaleState(NamedTuple):
    """Empty state for constant step-size scaling transformations."""


OptState = Union[
    EmptyState,
    TraceState,
    ScaleByScheduleState,
    ScaleState,
    NamedTupleProtocol,
    tuple["OptState", ...],
    list["OptState"],
    Mapping[str, "OptState"],
    PyTree,
]


class TransformInitFn(Protocol):
    """Protocol for a callable that initializes optimizer state from model parameters."""

    def __call__(self, params: Params) -> OptState:
        """Initialize and return the optimizer state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized optimizer state.
        """
        ...


class TransformUpdateFn(Protocol):
    """Protocol for a callable that transforms gradients and updates optimizer state."""

    def __call__(
        self,
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, OptState]:
        """Transform gradient `updates` and advance `state`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional current model parameters PyTree.

        Returns:
            A tuple `(transformed_updates, new_state)`.
        """
        ...


class TransformUpdateExtraArgsFn(Protocol):
    """Protocol for an update function accepting extra keyword arguments."""

    def __call__(
        self,
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: float | int | str | bool | Tensor | None,
    ) -> tuple[Updates, OptState]:
        """Transform gradient `updates` with optional extra keyword arguments.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional current model parameters PyTree.
            **extra_args: Additional keyword arguments passed to the transformation.

        Returns:
            A tuple `(transformed_updates, new_state)`.
        """
        ...


class GradientTransformation(NamedTuple):
    """A pair of pure functions `(init, update)` implementing a gradient transformation.

    Attributes:
        init: Pure function initializing the optimizer state from parameters.
        update: Pure function transforming updates and advancing the optimizer state.
    """

    init: TransformInitFn
    update: TransformUpdateFn


class GradientTransformationExtraArgs(NamedTuple):
    """Gradient transformation whose `update` accepts extra keyword arguments.

    Attributes:
        init: Pure function initializing the optimizer state from parameters.
        update: Pure function accepting extra keyword arguments during update.
    """

    init: TransformInitFn
    update: TransformUpdateExtraArgsFn


def identity() -> GradientTransformation:
    """Stateless identity transformation that leaves gradient updates unchanged.

    Returns:
        A `GradientTransformation` that returns updates unmodified.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for identity transformation.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            An `EmptyState` instance.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Return `updates` unchanged with an empty state.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(updates, EmptyState())`.
        """
        del state, params
        return updates, EmptyState()

    return GradientTransformation(init_fn, update_fn)


def set_to_zero() -> GradientTransformation:
    """Stateless transformation that sets all gradient updates to zero.

    Returns:
        A `GradientTransformation` that zeroes out all updates.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for set_to_zero transformation.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            An `EmptyState` instance.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Replace all leaves in `updates` with zeros of matching shape and dtype.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(zero_updates, EmptyState())`.
        """
        del state, params
        return tree_zeros_like(updates), EmptyState()

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "Array",
    "EmptyState",
    "GradientTransformation",
    "GradientTransformationExtraArgs",
    "OptState",
    "Params",
    "PyTree",
    "ScalarOrSchedule",
    "ScaleByScheduleState",
    "ScaleState",
    "Schedule",
    "TraceState",
    "TransformInitFn",
    "TransformUpdateExtraArgsFn",
    "TransformUpdateFn",
    "Updates",
    "identity",
    "set_to_zero",
]
