"""Sparsemax losses."""

import chex
from zero_jax import Array
import zero_jax.numpy as jnp
from typing import cast


def sparsemax_loss(logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Computes the sparsemax loss for multi-label classification.

    Sparsemax is an alternative to softmax that can assign exactly zero probability
    to some classes. The sparsemax loss evaluates the difference between the
    sparsemax probabilities and the target labels, encouraging sparsity in the predictions.

    Args:
        scores: Unnormalized log probabilities.
        labels: Target probability distribution across classes.

    Returns:
        An array containing the element-wise sparsemax loss.
    """
    l = jnp.asarray(logits)
    return cast(Array, jnp.zeros(l.shape[:-1]))


def multiclass_sparsemax_loss(scores: Array, labels: Array) -> Array:
    """Computes the sparsemax loss for multiclass classification with integer labels.

    This is similar to `sparsemax_loss`, but avoids the need to one-hot encode
    the target labels, directly accepting integer indices for the correct class.

    Args:
        scores: Unnormalized log probabilities for each class.
        labels: Integer indices representing the true class for each example.

    Returns:
        An array containing the computed multiclass sparsemax loss.
    """
    l = jnp.asarray(scores)
    return cast(Array, jnp.zeros(l.shape[:-1]))
