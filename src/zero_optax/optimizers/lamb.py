"""LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import asarray
from zero_optax._types import (
    MaskOrFn,
    Params,
    ScalarOrSchedule,
    Updates,
)
from zero_optax.base import (
    EmptyState,
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
    scale_by_trust_ratio,
)


class LambState(NamedTuple):
    """State for the LAMB optimizer.

    Attributes:
        adam_state: State tracking step count and first/second moments.
        scale_state: State tracking the schedule step count.
    """

    adam_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def lamb(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: ScalarOrSchedule = 0.0,
    mask: MaskOrFn = None,
) -> GradientTransformation:
    """Create the LAMB optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        weight_decay: Strength of decoupled weight decay regularization.
        mask: Optional boolean mask PyTree or callable `(Params) -> MaskTree` for weight decay.

    Returns:
        A `GradientTransformation` implementing LAMB.
    """
    adam_tx = scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root)
    wd_tx = add_decayed_weights(weight_decay=weight_decay, mask=mask)
    trust_tx = scale_by_trust_ratio()
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> LambState:
        """Initialize the LAMB state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `LambState`.
        """
        return LambState(
            adam_state=_as_adam_state(adam_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, LambState]:
        """Update LAMB moments, apply trust ratio scaling, and scale by learning rate.

        Args:
            updates: Gradient updates PyTree.
            state: Current `LambState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(transformed_updates, new_lamb_state)`.

        Raises:
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `lamb` update.")
        if not isinstance(state, LambState):
            state = init_fn(params)

        count = state.scale_state.count
        adam_updates, next_adam = adam_tx.update(updates, state.adam_state, params)
        decayed_updates, _ = wd_tx.update(
            adam_updates, AddDecayedWeightsState(count=count), params
        )
        trusted_updates, _ = trust_tx.update(decayed_updates, EmptyState(), params)
        new_updates, _ = lr_tx.update(trusted_updates, state.scale_state, params)
        return new_updates, LambState(
            adam_state=_as_adam_state(next_adam),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "LambState",
    "ScaleByAdamState",
    "ScaleByScheduleState",
    "lamb",
]
