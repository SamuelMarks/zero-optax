"""NovoGrad (Layer-wise Adaptive Moments) optimizer."""

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
    ScaleByNovogradState,
    scale_by_learning_rate,
    scale_by_novograd,
)


class NovogradState(NamedTuple):
    """State for the NovoGrad optimizer.

    Attributes:
        novograd_state: State tracking step count, momentum, and per-leaf norm EMA.
        scale_state: State tracking the schedule step count.
    """

    novograd_state: ScaleByNovogradState
    scale_state: ScaleByScheduleState


def _as_novograd_state(state: OptState) -> ScaleByNovogradState:
    """Narrow `OptState` from `scale_by_novograd` to `ScaleByNovogradState`.

    Args:
        state: State returned by `scale_by_novograd`.

    Returns:
        `ScaleByNovogradState` instance.

    Raises:
        TypeError: If `state` is not a `ScaleByNovogradState`.
    """
    if isinstance(state, ScaleByNovogradState):
        return state
    raise TypeError(f"Expected ScaleByNovogradState, got {type(state)!r}")


def novograd(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: ScalarOrSchedule = 0.0,
) -> GradientTransformation:
    """Create the NovoGrad optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Exponential decay rate for the first moment (momentum).
        b2: Exponential decay rate for the per-leaf squared gradient L2 norms.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        weight_decay: Strength of L2 weight decay added to normalized gradients.

    Returns:
        A `GradientTransformation` implementing NovoGrad.
    """
    novo_tx = scale_by_novograd(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
    )
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> NovogradState:
        """Initialize the NovoGrad state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `NovogradState`.
        """
        return NovogradState(
            novograd_state=_as_novograd_state(novo_tx.init(params)),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, NovogradState]:
        """Update NovoGrad layer-wise moments and scale by learning rate.

        Args:
            updates: Gradient updates PyTree.
            state: Current `NovogradState`.
            params: Optional model parameters PyTree (required when `weight_decay != 0`).

        Returns:
            Tuple of `(transformed_updates, new_novograd_state)`.

        Raises:
            ValueError: If `params` is None when `weight_decay` is non-zero.
        """
        if not isinstance(state, NovogradState):
            state = init_fn(updates)
        novo_updates, next_novo = novo_tx.update(updates, state.novograd_state, params)
        new_updates, _ = lr_tx.update(novo_updates, state.scale_state, params)
        return new_updates, NovogradState(
            novograd_state=_as_novograd_state(next_novo),
            scale_state=ScaleByScheduleState(count=state.scale_state.count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "NovogradState",
    "ScaleByNovogradState",
    "ScaleByScheduleState",
    "novograd",
]
