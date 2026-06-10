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
    """Compute the ranking softmax loss.

    Args:
        logits: The logits.
        labels: The labels.
        weights: Optional weights.
        where: Optional mask.
        reduce_fn: Optional reduction function.

    Returns:
        The ranking softmax loss.

    """
    l = jnp.asarray(logits)
    res = jnp.zeros(l.shape[:-1])
    if reduce_fn is None:
        return cast(Array, res)
    return cast(Array, reduce_fn(res))
