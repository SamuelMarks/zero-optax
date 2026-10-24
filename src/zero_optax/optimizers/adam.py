"""Adam optimizer."""

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
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
)
from zero_optax.transform import (
    ScaleByAdamState,
    scale_by_adam,
    scale_by_learning_rate,
)


class AdamState(NamedTuple):
    """State for the Adam optimizer.

    Attributes:
        adam_state: State tracking step count and first/second moments.
        scale_state: State tracking the schedule step count.
    """

    adam_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def _as_adam_state(state: OptState) -> ScaleByAdamState:
    """Narrow `OptState` from `scale_by_adam` to `ScaleByAdamState`.

    Args:
        state: State returned by `scale_by_adam` or `scale_by_yogi`.

    Returns:
        `ScaleByAdamState` instance.

    Raises:
        TypeError: If `state` is not a `ScaleByAdamState`.
    """
    if isinstance(state, ScaleByAdamState):
        return state
    raise TypeError(f"Expected ScaleByAdamState, got {type(state)!r}")


def adam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: DTypeLike = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation:
    """Create the Adam optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        mu_dtype: Optional data type for storing the first moment accumulator.
        nesterov: Whether to use Nesterov momentum for the first moment.

    Returns:
        A `GradientTransformation` implementing Adam.
    """
    adam_tx = scale_by_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
    )
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> AdamState:
        """Initialize the Adam state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `AdamState`.
        """
        return AdamState(
            adam_state=_as_adam_state(adam_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AdamState]:
        """Update Adam moments and compute bias-corrected parameter updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AdamState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_adam_state)`.
        """
        if not isinstance(state, AdamState):
            state = init_fn(updates)
        adam_updates, next_adam = adam_tx.update(updates, state.adam_state, params)
        new_updates, _ = lr_tx.update(adam_updates, state.scale_state, params)
        return new_updates, AdamState(
            adam_state=_as_adam_state(next_adam),
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "AdamState",
    "ScaleByAdamState",
    "ScaleByScheduleState",
    "adam",
]
