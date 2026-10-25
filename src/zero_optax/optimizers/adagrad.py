"""Adagrad optimizer."""

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
    ScaleByRssState,
    scale_by_learning_rate,
    scale_by_rss,
)


class AdagradState(NamedTuple):
    """State for the Adagrad optimizer.

    Attributes:
        rss_state: State tracking the sum of squared gradients.
        scale_state: State tracking the schedule step count.
    """

    rss_state: ScaleByRssState
    scale_state: ScaleByScheduleState


def adagrad(
    learning_rate: ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> GradientTransformation[AdagradState]:
    """Create the Adagrad optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        initial_accumulator_value: Starting value for the sum-of-squares accumulator.
        eps: Small constant added inside the square root for numerical stability.

    Returns:
        A `GradientTransformation` implementing Adagrad.
    """
    rss_tx = scale_by_rss(initial_accumulator_value=initial_accumulator_value, eps=eps)
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> AdagradState:
        """Initialize the Adagrad state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `AdagradState`.
        """
        return AdagradState(
            rss_state=rss_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AdagradState]:
        """Update sum-of-squares accumulator and compute Adagrad parameter updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AdagradState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_adagrad_state)`.

        Raises:
            TypeError: If `state` is not an `AdagradState`.
        """
        if not isinstance(state, AdagradState):
            raise TypeError(f"Expected AdagradState in adagrad, got {type(state)!r}.")
        rss_updates, next_rss = rss_tx.update(updates, state.rss_state, params)
        new_updates, _ = lr_tx.update(rss_updates, state.scale_state, params)
        return new_updates, AdagradState(
            rss_state=next_rss,
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "AdagradState",
    "adagrad",
]
