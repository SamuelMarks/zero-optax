"""Sparsemax losses."""

from zero_jax import Array
import zero_jax.numpy as jnp
from typing import cast


def sparsemax_loss(logits: Array, labels: Array) -> Array:
    """Compute the sparsemax loss.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The sparsemax loss.

    """
    l = jnp.asarray(logits)
    return cast(Array, jnp.zeros(l.shape[:-1]))


def multiclass_sparsemax_loss(logits: Array, labels: Array) -> Array:
    """Compute the multiclass sparsemax loss.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The multiclass sparsemax loss.

    """
    l = jnp.asarray(logits)
    return cast(Array, jnp.zeros(l.shape[:-1]))
