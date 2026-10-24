"""AdamW optimizer (Adam with decoupled weight decay)."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import asarray
from zero_optax._types import (
    DTypeLike,
    MaskOrFn,
    Params,
    ScalarOrSchedule,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
)
from zero_optax.optimizers.adam import _as_adam_state
from zero_optax.transform import (
    AddDecayedWeightsState,
    ScaleByAdamState,
    add_decayed_weights,
    scale_by_adam,
    scale_by_learning_rate,
)


class AdamwState(NamedTuple):
    """State for the AdamW optimizer.

    Attributes:
        adam_state: State tracking step count and first/second moments.
        scale_state: State tracking the schedule step count.
    """

    adam_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def adamw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: DTypeLike = None,
    weight_decay: ScalarOrSchedule = 1e-4,
    mask: MaskOrFn = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation:
    """Create the AdamW optimizer (Adam with decoupled weight decay).

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        mu_dtype: Optional data type for storing the first moment accumulator.
        weight_decay: Strength of decoupled weight decay regularization.
        mask: Optional boolean mask PyTree or callable `(Params) -> MaskTree` for weight decay.
        nesterov: Whether to use Nesterov momentum for the first moment.

    Returns:
        A `GradientTransformation` implementing AdamW.
    """
    adam_tx = scale_by_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
    )
    wd_tx = add_decayed_weights(weight_decay=weight_decay, mask=mask)
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> AdamwState:
        """Initialize the AdamW state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `AdamwState`.
        """
        return AdamwState(
            adam_state=_as_adam_state(adam_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AdamwState]:
        """Update AdamW moments, apply decoupled weight decay, and scale by learning rate.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AdamwState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(transformed_updates, new_adamw_state)`.

        Raises:
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `adamw` update.")
        if not isinstance(state, AdamwState):
            state = init_fn(params)

        count = state.scale_state.count
        adam_updates, next_adam = adam_tx.update(updates, state.adam_state, params)
        decayed_updates, _ = wd_tx.update(
            adam_updates, AddDecayedWeightsState(count=count), params
        )
        new_updates, _ = lr_tx.update(decayed_updates, state.scale_state, params)
        return new_updates, AdamwState(
            adam_state=_as_adam_state(next_adam),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "AdamwState",
    "ScaleByAdamState",
    "ScaleByScheduleState",
    "adamw",
]
