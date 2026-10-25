"""Regression loss functions."""

from __future__ import annotations

import math

import zero_chex

from zero_optax._tree import (
    abs_tensor,
    asarray,
    exp,
    log1p,
    maximum,
    reduce_sum,
    sqrt,
    zeros_like,
)
from zero_optax._tree import (
    where as where_select,
)
from zero_optax._types import ArrayLike, Tensor


def squared_error(
    predictions: ArrayLike,
    targets: ArrayLike | None = None,
) -> Tensor:
    """Calculate the element-wise squared error for a set of predictions.

    Args:
        predictions: Predicted values tensor of arbitrary shape.
        targets: Ground-truth target values matching `predictions`. If
            None, targets are assumed to be zeros matching `predictions`.

    Returns:
        Element-wise squared error `(predictions - targets) ** 2`.
    """
    p = asarray(predictions)
    if targets is None:
        t = zeros_like(p)
    else:
        t = asarray(targets)
        zero_chex.assert_equal_shape([p, t])
    diff = p - t
    return diff * diff


def l2_loss(
    predictions: ArrayLike,
    targets: ArrayLike | None = None,
) -> Tensor:
    """Calculate the L2 loss (half squared error) for a set of predictions.

    Args:
        predictions: Predicted values tensor of arbitrary shape.
        targets: Ground-truth target values broadcastable to `predictions`. If
            None, targets are assumed to be zeros matching `predictions`.

    Returns:
        Element-wise L2 loss `0.5 * (predictions - targets) ** 2`.
    """
    return 0.5 * squared_error(predictions, targets)


def huber_loss(
    predictions: ArrayLike,
    targets: ArrayLike | None = None,
    *,
    delta: float | ArrayLike = 1.0,
) -> Tensor:
    """Calculate the Huber loss for a set of predictions.

    Combines quadratic loss for small errors (`|diff| < delta`) and linear
    loss for large errors (`|diff| >= delta`).

    Args:
        predictions: Predicted values tensor of arbitrary shape.
        targets: Ground-truth target values broadcastable to `predictions`. If
            None, targets are assumed to be zeros matching `predictions`.
        delta: Threshold radius separating quadratic and linear error regions.

    Returns:
        Element-wise Huber loss Tensor.
    """
    p = asarray(predictions)
    t = zeros_like(p) if targets is None else asarray(targets)
    d = asarray(delta)
    diff = abs_tensor(p - t)
    return where_select(diff < d, 0.5 * diff * diff, d * (diff - 0.5 * d))


def log_cosh(
    predictions: ArrayLike,
    targets: ArrayLike | None = None,
) -> Tensor:
    """Calculate the log-cosh loss for a set of predictions.

    `log(cosh(x))` is a twice-differentiable smooth approximation to Huber loss,
    behaving like `x^2 / 2` for small `x` and `|x| - log(2)` for large `x`.

    Args:
        predictions: Predicted values tensor of arbitrary shape.
        targets: Ground-truth target values broadcastable to `predictions`. If
            None, targets are assumed to be zeros matching `predictions`.

    Returns:
        Element-wise log-cosh loss Tensor.
    """
    p = asarray(predictions)
    errors = p if targets is None else p - asarray(targets)
    abs_err = abs_tensor(errors)
    return abs_err + log1p(exp(-2.0 * abs_err)) - math.log(2.0)


def cosine_similarity(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    epsilon: float = 0.0,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute the cosine similarity between `predictions` and `targets`.

    Args:
        predictions: Predicted vectors of shape `[..., dim]`.
        targets: Ground-truth target vectors of shape `[..., dim]`.
        epsilon: Minimum norm floor for denominator terms to avoid division by zero.
        axis: Axis or axes along which the vector inner product is computed.
        where: Optional boolean mask specifying elements to include.

    Returns:
        Cosine similarity Tensor with `axis` reduced.
    """
    a = asarray(predictions)
    b = asarray(targets)
    if where is not None:
        w = asarray(where)
        a = where_select(w, a, 0.0)
        b = where_select(w, b, 0.0)
    a_norm2 = reduce_sum(a * a, axis=axis, keepdims=True)
    b_norm2 = reduce_sum(b * b, axis=axis, keepdims=True)
    eps_t = asarray(epsilon)
    a_norm = sqrt(maximum(a_norm2, eps_t))
    b_norm = sqrt(maximum(b_norm2, eps_t))
    a_unit = where_select(
        a_norm > 0.0, a / where_select(a_norm > 0.0, a_norm, 1.0), 0.0
    )
    b_unit = where_select(
        b_norm > 0.0, b / where_select(b_norm > 0.0, b_norm, 1.0), 0.0
    )
    prod = a_unit * b_unit
    if where is not None:
        prod = where_select(asarray(where), prod, 0.0)
    return reduce_sum(prod, axis=axis)


def cosine_distance(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    epsilon: float = 0.0,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute the cosine distance `1.0 - cosine_similarity(predictions, targets)`.

    Args:
        predictions: Predicted vectors of shape `[..., dim]`.
        targets: Ground-truth target vectors of shape `[..., dim]`.
        epsilon: Minimum norm floor for denominator terms.
        axis: Axis or axes along which to compute distance.
        where: Optional boolean mask specifying elements to include.

    Returns:
        Cosine distance Tensor with `axis` reduced.
    """
    return 1.0 - cosine_similarity(
        predictions, targets, epsilon=epsilon, axis=axis, where=where
    )


__all__ = [
    "cosine_distance",
    "cosine_similarity",
    "huber_loss",
    "l2_loss",
    "log_cosh",
    "squared_error",
]
