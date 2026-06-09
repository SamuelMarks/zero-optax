"""Ranking loss."""

import zero_jax.numpy as jnp
from typing import Optional, Union
import collections.abc
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]


def ranking_softmax_loss(
    logits: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    weights: Optional[Array] = None,
    reduce_fn: Union["collections.abc.Callable[..., Array]", None] = jnp.mean,
) -> Array:
    """Ranking softmax loss."""
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp_logits

    loss = -np.sum(labels * np.log(probs + 1e-10), axis=-1)

    if weights is not None:
        loss = loss * weights

    if where is not None:
        loss = np.where(where, loss, 0.0)

    loss = jnp.array(loss)
    if reduce_fn is not None:
        return reduce_fn(loss)
    return loss
