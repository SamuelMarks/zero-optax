"""Yogi adaptive optimizer."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import asarray
from zero_optax._types import Params, ScalarOrSchedule, Updates
from zero_optax.base import (
    GradientTransformation,
    OptState,
    ScaleByScheduleState,
)
from zero_optax.transform import (
    ScaleByAdamState,
    scale_by_learning_rate,
    scale_by_yogi,
)


class YogiState(NamedTuple):
    """State for the Yogi optimizer.

    Attributes:
        yogi_state: State tracking step count and first/second moments.
        scale_state: State tracking the schedule step count.
    """

    yogi_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def yogi(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
) -> GradientTransformation[YogiState]:
    """Create the Yogi adaptive optimizer.

    Yogi controls effective learning rate growth using an additive second
    moment update `v_t = v_{t-1} - (1 - b2) * sign(v_{t-1} - g_t^2) * g_t^2`.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment estimate.
        b2: Adaptation rate for the additive second moment estimate.
        eps: Small constant added outside the square root in the denominator.

    Returns:
        A `GradientTransformation` implementing Yogi.
    """
    yogi_tx = scale_by_yogi(
        b1=b1,
        b2=b2,
        eps=eps,
        initial_accumulator_value=1e-6,
    )
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> YogiState:
        """Initialize the Yogi state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `YogiState`.
        """
        return YogiState(
            yogi_state=yogi_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, YogiState]:
        """Update Yogi moments and compute bias-corrected parameter updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `YogiState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_yogi_state)`.

        Raises:
            TypeError: If `state` is not a `YogiState`.
        """
        if not isinstance(state, YogiState):
            raise TypeError(f"Expected YogiState in yogi, got {type(state)!r}.")
        yogi_updates, next_yogi = yogi_tx.update(updates, state.yogi_state, params)
        new_updates, _ = lr_tx.update(yogi_updates, state.scale_state, params)
        return new_updates, YogiState(
            yogi_state=next_yogi,
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "YogiState",
    "yogi",
]
