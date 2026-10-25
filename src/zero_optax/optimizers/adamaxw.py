"""AdaMaxW optimizer (AdaMax with decoupled weight decay)."""

from __future__ import annotations

from zero_optax._types import (
    MaskOrFn,
    ScalarOrSchedule,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
)
from zero_optax.combine import chain
from zero_optax.optimizers.adamax import scale_by_adamax
from zero_optax.transform import add_decayed_weights, scale_by_learning_rate


def adamaxw(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: ScalarOrSchedule = 1e-4,
    mask: MaskOrFn = None,
) -> GradientTransformation[tuple[OptState, ...]]:
    """AdaMax with decoupled weight decay regularization.

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the exponentially weighted infinity norm.
        eps: Small constant added to denominator for numerical stability.
        weight_decay: Strength of decoupled weight decay regularization.
        mask: Optional boolean mask PyTree or callable `mask(params)` for weight decay.

    Returns:
        A `GradientTransformation` implementing AdaMaxW.
    """
    return chain(
        scale_by_adamax(b1=b1, b2=b2, eps=eps),
        add_decayed_weights(weight_decay=weight_decay, mask=mask),
        scale_by_learning_rate(learning_rate),
    )


__all__ = ["adamaxw"]
