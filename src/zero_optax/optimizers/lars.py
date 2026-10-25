"""LARS (Layer-wise Adaptive Rate Scaling) optimizer."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import apply_mask_tree, asarray, resolve_mask
from zero_optax._types import (
    MaskOrFn,
    Params,
    ScalarOrSchedule,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
    TraceState,
)
from zero_optax.transform import (
    AddDecayedWeightsState,
    ScaleByTrustRatioState,
    add_decayed_weights,
    scale_by_learning_rate,
    scale_by_trust_ratio,
    trace,
)


class LarsState(NamedTuple):
    """State for the LARS optimizer.

    Attributes:
        trace_state: State holding momentum trace.
        scale_state: State tracking the schedule step count.
    """

    trace_state: TraceState
    scale_state: ScaleByScheduleState


def lars(
    learning_rate: ScalarOrSchedule,
    weight_decay: float = 0.0,
    weight_decay_mask: MaskOrFn = True,
    trust_coefficient: float = 0.001,
    eps: float = 0.0,
    trust_ratio_mask: MaskOrFn = True,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> GradientTransformation[LarsState]:
    """Create the LARS (Layer-wise Adaptive Rate Scaling) optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        weight_decay: Strength of L2 weight decay regularization.
        weight_decay_mask: Mask or callable specifying which parameters get weight decay.
        trust_coefficient: Trust coefficient multiplier (`eeta`).
        eps: Additive constant in the trust ratio denominator.
        trust_ratio_mask: Mask or callable specifying which parameters get trust ratio scaling.
        momentum: Momentum decay rate.
        nesterov: Whether to apply Nesterov momentum.

    Returns:
        A `GradientTransformation` implementing LARS.
    """
    wd_tx = add_decayed_weights(weight_decay=weight_decay, mask=weight_decay_mask)
    trust_tx = scale_by_trust_ratio(
        trust_coefficient=trust_coefficient,
        eps=eps,
    )
    lr_tx = scale_by_learning_rate(learning_rate)
    trace_tx = trace(decay=momentum, nesterov=nesterov)

    def init_fn(params: Params) -> LarsState:
        """Initialize the LARS state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `LarsState`.
        """
        return LarsState(
            trace_state=trace_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, LarsState]:
        """Apply weight decay, trust-ratio scaling, learning rate, and momentum.

        Args:
            updates: Gradient updates PyTree.
            state: Current `LarsState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(transformed_updates, new_lars_state)`.

        Raises:
            TypeError: If `state` is not a `LarsState`.
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `lars` update.")
        if not isinstance(state, LarsState):
            raise TypeError(f"Expected LarsState in lars, got {type(state)!r}.")

        count = state.scale_state.count
        decayed_updates, _ = wd_tx.update(
            updates, AddDecayedWeightsState(count=count), params
        )
        scaled_updates, _ = trust_tx.update(
            decayed_updates, ScaleByTrustRatioState(), params
        )
        tr_mask_tree = resolve_mask(trust_ratio_mask, params)
        if tr_mask_tree is not None:
            scaled_updates = apply_mask_tree(
                scaled_updates, decayed_updates, tr_mask_tree
            )
        lr_updates, _ = lr_tx.update(scaled_updates, state.scale_state, params)
        final_updates, next_trace = trace_tx.update(
            lr_updates, state.trace_state, params
        )
        return final_updates, LarsState(
            trace_state=next_trace,
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "LarsState",
    "lars",
]
