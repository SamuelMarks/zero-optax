"""Sparsemax losses."""

import zero_jax.numpy as jnp
from typing import Union
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]


def _sparsemax_projection(z: Array) -> Array:
    """Compute the sparsemax projection of z."""
    # Sort z in descending order
    z_sorted = np.sort(z, axis=-1)[..., ::-1]
    # Calculate cumulative sum
    z_cumsum = np.cumsum(z_sorted, axis=-1)
    k = np.arange(1, z.shape[-1] + 1)

    # Find the support size
    support = 1 + k * z_sorted > z_cumsum
    k_z = np.sum(support, axis=-1, keepdims=True)

    # Calculate tau
    tau_z = (np.take_along_axis(z_cumsum, k_z - 1, axis=-1) - 1) / k_z
    return np.maximum(0.0, z - tau_z)


def sparsemax_loss(logits: Array, labels: Array) -> Array:
    """Binary sparsemax loss."""
    p = _sparsemax_projection(logits)
    loss = (
        -np.sum(logits * labels, axis=-1)
        + 0.5 * np.sum(p**2, axis=-1)
        + 0.5 * np.sum(labels**2, axis=-1)
    )
    return jnp.array(loss)


def multiclass_sparsemax_loss(scores: Array, labels: Array) -> Array:
    """Multiclass sparsemax loss."""
    labels_one_hot = jax.nn.one_hot(labels, scores.shape[-1])
    return sparsemax_loss(scores, labels_one_hot)
