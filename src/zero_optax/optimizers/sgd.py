"""Stochastic Gradient Descent (SGD) optimizer."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import asarray
from zero_optax._types import (
    DTypeLike,
    Params,
    ScalarOrSchedule,
    Updates,
)
from zero_optax.base import (
    EmptyState,
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
    TraceState,
    identity,
)
from zero_optax.optimizers.rmsprop import _as_trace_or_empty
from zero_optax.transform import scale_by_learning_rate, trace


class SgdState(NamedTuple):
    """State for the SGD optimizer.

    Attributes:
        trace_state: `TraceState` when momentum is enabled, otherwise `EmptyState`.
        scale_state: State tracking the schedule step count.
    """

    trace_state: TraceState | EmptyState
    scale_state: ScaleByScheduleState


def sgd(
    learning_rate: ScalarOrSchedule,
    momentum: float | None = None,
    nesterov: bool = False,
    accumulator_dtype: DTypeLike = None,
) -> GradientTransformation:
    """Create a canonical Stochastic Gradient Descent (SGD) optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        momentum: Optional decay rate for the momentum trace. If None, no momentum is used.
        nesterov: Whether to apply Nesterov momentum lookahead.
        accumulator_dtype: Optional data type for the momentum trace accumulator.

    Returns:
        A `GradientTransformation` implementing SGD.
    """
    trace_tx = (
        trace(decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype)
        if momentum is not None and momentum != 0.0
        else identity()
    )
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> SgdState:
        """Initialize the SGD state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `SgdState`.
        """
        return SgdState(
            trace_state=_as_trace_or_empty(trace_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, SgdState]:
        """Apply optional momentum and learning rate scaling to `updates`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `SgdState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_sgd_state)`.
        """
        if not isinstance(state, SgdState):
            state = init_fn(updates)
        step_dir, next_t = trace_tx.update(updates, state.trace_state, params)
        new_updates, _ = lr_tx.update(step_dir, state.scale_state, params)
        return new_updates, SgdState(
            trace_state=_as_trace_or_empty(next_t),
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "EmptyState",
    "ScaleByScheduleState",
    "SgdState",
    "TraceState",
    "sgd",
]
