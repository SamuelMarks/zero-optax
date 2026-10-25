"""RMSProp optimizer."""

from __future__ import annotations

from typing import NamedTuple, Union

from zero_optax._tree import asarray
from zero_optax._types import Params, ScalarOrSchedule, Updates
from zero_optax.base import (
    EmptyState,
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
    TraceState,
    identity,
)
from zero_optax.transform import (
    ScaleByRmsState,
    ScaleByRmsWithCountState,
    ScaleByRStdDevState,
    ScaleByRStdDevWithCountState,
    scale_by_learning_rate,
    scale_by_rms,
    scale_by_stddev,
    trace,
)

RmsInnerState = Union[
    ScaleByRmsState,
    ScaleByRmsWithCountState,
    ScaleByRStdDevState,
    ScaleByRStdDevWithCountState,
]


class RmsPropState(NamedTuple):
    """State for the RMSProp optimizer.

    Attributes:
        rms_state: State tracking squared gradients (and mean if `centered=True`).
        scale_state: State tracking the schedule step count.
        trace_state: `TraceState` when `momentum` is set, otherwise `EmptyState`.
    """

    rms_state: RmsInnerState
    scale_state: ScaleByScheduleState
    trace_state: TraceState | EmptyState


def rmsprop(
    learning_rate: ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    centered: bool = False,
    momentum: float | None = None,
    nesterov: bool = False,
    bias_correction: bool = False,
) -> GradientTransformation[RmsPropState]:
    """Create a flexible RMSProp optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        decay: Exponential decay rate for tracking gradient magnitude.
        eps: Small numerical constant to prevent division by zero.
        initial_scale: Initial value of the squared gradient accumulator.
        eps_in_sqrt: Whether `eps` is added inside `sqrt(v + eps)` or outside `sqrt(v) + eps`.
        centered: If True, rescale by centered variance `nu - mu^2` instead of `nu`.
        momentum: Optional momentum decay rate. If None, momentum is disabled.
        nesterov: Whether Nesterov momentum is used when `momentum` is not None.
        bias_correction: Whether to apply `(1 - decay^t)` bias correction to moment estimates.

    Returns:
        A `GradientTransformation` implementing RMSProp.
    """
    rms_tx: GradientTransformation[RmsInnerState] = (
        scale_by_stddev(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            bias_correction=bias_correction,
        )
        if centered
        else scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            bias_correction=bias_correction,
        )
    )
    lr_tx = scale_by_learning_rate(learning_rate)
    trace_tx: GradientTransformation[TraceState | EmptyState] = (
        trace(decay=momentum, nesterov=nesterov) if momentum is not None else identity()
    )

    def init_fn(params: Params) -> RmsPropState:
        """Initialize the RMSProp state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `RmsPropState`.
        """
        return RmsPropState(
            rms_state=rms_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
            trace_state=trace_tx.init(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, RmsPropState]:
        """Update RMSProp accumulators, apply learning rate, and apply optional momentum.

        Args:
            updates: Gradient updates PyTree.
            state: Current `RmsPropState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_rmsprop_state)`.

        Raises:
            TypeError: If `state` is not an `RmsPropState`.
        """
        if not isinstance(state, RmsPropState):
            raise TypeError(f"Expected RmsPropState in rmsprop, got {type(state)!r}.")

        scaled_grads, next_rms = rms_tx.update(updates, state.rms_state, params)
        lr_updates, _ = lr_tx.update(scaled_grads, state.scale_state, params)
        final_updates, next_trace = trace_tx.update(
            lr_updates, state.trace_state, params
        )
        return final_updates, RmsPropState(
            rms_state=next_rms,
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
            trace_state=next_trace,
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "RmsPropState",
    "rmsprop",
]
