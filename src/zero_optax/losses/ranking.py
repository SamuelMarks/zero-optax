"""Ranking losses."""

from typing import Any, Callable, Optional

import chex
import numpy as np


def ranking_softmax_loss(
    logits: chex.Array,
    labels: chex.Array,
    weights: Optional[chex.Array] = None,
    where: Optional[chex.Array] = None,
    reduce_fn: Optional[Callable[..., Any]] = np.mean,
) -> chex.Array:
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
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        res = np.zeros(l.shape[:-1])
        if reduce_fn is None:
            return res
        return reduce_fn(res)
    return logits
