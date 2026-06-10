"""Regression losses."""

from typing import Any, Optional

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp


def squared_error(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Compute the squared error.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).

    Returns:
        The squared error.

    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    return cast(Array, (p - t) ** 2)


def l2_loss(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Compute the L2 loss.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).

    Returns:
        The L2 loss.

    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    return cast(Array, 0.5 * (p - t) ** 2)


def huber_loss(
    predictions: Array, targets: Optional[Array] = None, delta: float = 1.0
) -> Array:
    """Compute the Huber loss.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).
        delta: The threshold at which to change from a squared error to an absolute error.

    Returns:
        The Huber loss.

    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    diff = jnp.abs(p - t)
    return cast(
        Array, jnp.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    )
