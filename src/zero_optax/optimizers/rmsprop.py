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


def _as_rms_inner_state(state: OptState) -> RmsInnerState:
    """Narrow `OptState` from `scale_by_rms` / `scale_by_stddev` to `RmsInnerState`.

    Args:
        state: State returned by `scale_by_rms` or `scale_by_stddev`.

    Returns:
        `RmsInnerState` instance.

    Raises:
        TypeError: If `state` is not a valid `RmsInnerState`.
    """
    if isinstance(
        state,
        (
            ScaleByRmsState,
            ScaleByRmsWithCountState,
            ScaleByRStdDevState,
            ScaleByRStdDevWithCountState,
        ),
    ):
        return state
    raise TypeError(f"Expected RmsInnerState, got {type(state)!r}")


def _as_trace_or_empty(state: OptState) -> TraceState | EmptyState:
    """Narrow `OptState` from `trace` / `identity` to `TraceState | EmptyState`.

    Args:
        state: State returned by `trace` or `identity`.

    Returns:
        `TraceState` or `EmptyState` instance.

    Raises:
        TypeError: If `state` is neither `TraceState` nor `EmptyState`.
    """
    if isinstance(state, (TraceState, EmptyState)):
        return state
    raise TypeError(f"Expected TraceState or EmptyState, got {type(state)!r}")


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
) -> GradientTransformation:
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
    rms_tx = (
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
    trace_tx = (
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
            rms_state=_as_rms_inner_state(rms_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
            trace_state=_as_trace_or_empty(trace_tx.init(params)),
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
        """
        if not isinstance(state, RmsPropState):
            state = init_fn(updates)

        scaled_grads, next_rms = rms_tx.update(updates, state.rms_state, params)
        lr_updates, _ = lr_tx.update(scaled_grads, state.scale_state, params)
        final_updates, next_trace = trace_tx.update(
            lr_updates, state.trace_state, params
        )
        return final_updates, RmsPropState(
            rms_state=_as_rms_inner_state(next_rms),
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
            trace_state=_as_trace_or_empty(next_trace),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "RmsPropState",
    "ScaleByRStdDevState",
    "ScaleByRStdDevWithCountState",
    "ScaleByRmsState",
    "ScaleByRmsWithCountState",
    "rmsprop",
]
