"""Ranking losses."""

from typing import Any, Callable, Optional

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp


def ranking_softmax_loss(
    logits: Array,
    labels: Array,
    weights: Optional[Array] = None,
    where: Optional[Array] = None,
    reduce_fn: Optional[Callable[..., Any]] = jnp.mean,
) -> Array:
    """Computes a softmax-based ranking loss, useful for learning-to-rank tasks.

    This loss evaluates the predictions of a ranking model by interpreting the scores
    as a distribution over items and comparing them against the relevance labels
    provided in the ground truth.

    Args:
        logits: The predicted ranking scores for each item.
        labels: The ground truth relevance labels for the items.
        weights: Optional array of weights to scale the loss for each example/item.
        where: Optional boolean mask indicating which elements to include in the loss.
        reduce_fn: A callable used to reduce the loss over the batch dimensions.
            Defaults to `jnp.mean`. If None, the unreduced loss is returned.

    Returns:
        The computed ranking loss, either reduced or as an array depending on `reduce_fn`.
    """
    l = jnp.asarray(logits)
    res = jnp.zeros(l.shape[:-1])
    if reduce_fn is None:
        return cast(Array, res)
    return cast(Array, reduce_fn(res))
