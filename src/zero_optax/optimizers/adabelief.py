"""AdaBelief optimizer."""

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
    ScaleByBeliefState,
    scale_by_belief,
    scale_by_learning_rate,
)


class AdaBeliefState(NamedTuple):
    """State for the AdaBelief optimizer.

    Attributes:
        belief_state: State tracking step count, first moment, and central second moment.
        scale_state: State tracking the schedule step count.
    """

    belief_state: ScaleByBeliefState
    scale_state: ScaleByScheduleState


def adabelief(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    *,
    nesterov: bool = False,
) -> GradientTransformation[AdaBeliefState]:
    """Create the AdaBelief optimizer.

    Adapts step sizes according to the "belief" in observed gradients via the
    exponentially moving variance `(g_t - m_t)^2`.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second central moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added to the prediction error variance at each step.
        nesterov: Whether to apply Nesterov momentum to the first moment estimate.

    Returns:
        A `GradientTransformation` implementing AdaBelief.
    """
    belief_tx = scale_by_belief(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        nesterov=nesterov,
    )
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> AdaBeliefState:
        """Initialize the AdaBelief state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `AdaBeliefState`.
        """
        return AdaBeliefState(
            belief_state=belief_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AdaBeliefState]:
        """Update AdaBelief moments and compute parameter updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AdaBeliefState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(transformed_updates, new_adabelief_state)`.

        Raises:
            TypeError: If `state` is not an `AdaBeliefState`.
        """
        if not isinstance(state, AdaBeliefState):
            raise TypeError(
                f"Expected AdaBeliefState in adabelief, got {type(state)!r}."
            )
        belief_updates, next_belief = belief_tx.update(
            updates, state.belief_state, params
        )
        new_updates, _ = lr_tx.update(belief_updates, state.scale_state, params)
        return new_updates, AdaBeliefState(
            belief_state=next_belief,
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "AdaBeliefState",
    "adabelief",
]
