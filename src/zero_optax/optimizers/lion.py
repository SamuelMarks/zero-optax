"""Lion (EvoLved Sign Momentum) optimizer."""

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
from zero_optax.transform import (
    AddDecayedWeightsState,
    ScaleByLionState,
    add_decayed_weights,
    scale_by_learning_rate,
    scale_by_lion,
)


class LionState(NamedTuple):
    """State for the Lion optimizer.

    Attributes:
        lion_state: State holding step count and momentum trace.
        scale_state: State tracking the schedule step count.
    """

    lion_state: ScaleByLionState
    scale_state: ScaleByScheduleState


def lion(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: DTypeLike = None,
    weight_decay: ScalarOrSchedule = 1e-3,
    mask: MaskOrFn = None,
) -> GradientTransformation[LionState]:
    """Create the Lion (EvoLved Sign Momentum) optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        b1: Interpolation rate for computing the sign update direction.
        b2: Exponential decay rate for updating the momentum trace.
        mu_dtype: Optional data type for storing the momentum accumulator.
        weight_decay: Strength of decoupled weight decay regularization.
        mask: Optional boolean mask PyTree or callable `(Params) -> MaskTree` for weight decay.

    Returns:
        A `GradientTransformation` implementing Lion.
    """
    lion_tx = scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype)
    wd_tx = add_decayed_weights(weight_decay=weight_decay, mask=mask)
    lr_tx = scale_by_learning_rate(learning_rate)

    def init_fn(params: Params) -> LionState:
        """Initialize the Lion state for `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `LionState`.
        """
        return LionState(
            lion_state=lion_tx.init(params),
            scale_state=ScaleByScheduleState(count=asarray(0)),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, LionState]:
        """Compute Lion sign update, apply decoupled weight decay, and update momentum.

        Args:
            updates: Gradient updates PyTree.
            state: Current `LionState`.
            params: Current model parameters PyTree (required when weight decay is active).

        Returns:
            Tuple of `(transformed_updates, new_lion_state)`.

        Raises:
            TypeError: If `state` is not a `LionState`.
            ValueError: If `params` is None.
        """
        if params is None and (callable(weight_decay) or weight_decay != 0.0):
            raise ValueError("`params` must be provided to `lion` update.")
        if not isinstance(state, LionState):
            raise TypeError(f"Expected LionState in lion, got {type(state)!r}.")

        count = state.scale_state.count
        sign_updates, next_lion = lion_tx.update(updates, state.lion_state, params)
        if params is not None:
            decayed_updates, _ = wd_tx.update(
                sign_updates, AddDecayedWeightsState(count=count), params
            )
        else:
            decayed_updates = sign_updates
        new_updates, _ = lr_tx.update(decayed_updates, state.scale_state, params)
        return new_updates, LionState(
            lion_state=next_lion,
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "LionState",
    "lion",
]
