"""Regression losses."""

import zero_jax.numpy as jnp
from typing import Optional, Union
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]


def squared_error(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the squared error for a set of predictions."""
    if targets is None:
        return predictions**2
    return (predictions - targets) ** 2  # type: ignore


def l2_loss(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the L2 loss for a set of predictions."""
    return 0.5 * squared_error(predictions, targets)


def huber_loss(
    predictions: Array, targets: Optional[Array] = None, delta: float = 1.0
) -> Array:
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero."""
    if targets is None:
        diff = predictions
    else:
        diff = predictions - targets  # type: ignore
    abs_diff = jnp.abs(diff)
    squared_loss = 0.5 * (diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    return jnp.where(abs_diff <= delta, squared_loss, linear_loss)
