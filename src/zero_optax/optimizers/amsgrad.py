"""AMSGrad optimizer."""

from __future__ import annotations

from zero_optax._types import (
    DTypeLike,
    ScalarOrSchedule,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
)
from zero_optax.combine import chain
from zero_optax.transform import scale_by_amsgrad, scale_by_learning_rate


def amsgrad(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: DTypeLike = None,
    bias_correction_mu: bool = True,
    bias_correction_nu: bool = True,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The AMSGrad optimizer.

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        mu_dtype: Optional dtype for the first moment accumulator `mu`.
        bias_correction_mu: Whether to apply bias correction to `mu`.
        bias_correction_nu: Whether to apply bias correction to `nu` before taking `nu_max`.

    Returns:
        A `GradientTransformation` implementing AMSGrad.
    """
    return chain(
        scale_by_amsgrad(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            bias_correction_mu=bias_correction_mu,
            bias_correction_nu=bias_correction_nu,
        ),
        scale_by_learning_rate(learning_rate),
    )


__all__ = ["amsgrad"]
