"""Sparsemax losses."""

import chex
import numpy as np


def sparsemax_loss(logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Compute the sparsemax loss.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The sparsemax loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        return np.zeros(l.shape[:-1])
    return logits


def multiclass_sparsemax_loss(logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Compute the multiclass sparsemax loss.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The multiclass sparsemax loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        return np.zeros(l.shape[:-1])
    return logits
