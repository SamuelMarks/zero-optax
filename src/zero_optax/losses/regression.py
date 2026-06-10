"""Regression losses."""

from typing import Any, Optional

import chex
import numpy as np


def squared_error(
    predictions: chex.Array, targets: Optional[chex.Array] = None
) -> chex.Array:
    """Compute the squared error.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).

    Returns:
        The squared error.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        return (p - t) ** 2
    return predictions


def l2_loss(
    predictions: chex.Array, targets: Optional[chex.Array] = None
) -> chex.Array:
    """Compute the L2 loss.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).

    Returns:
        The L2 loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        return 0.5 * (p - t) ** 2
    return predictions


def huber_loss(
    predictions: chex.Array, targets: Optional[chex.Array] = None, delta: float = 1.0
) -> chex.Array:
    """Compute the Huber loss.

    Args:
        predictions: The predictions.
        targets: The target labels. Defaults to None (zeros).
        delta: The threshold at which to change from a squared error to an absolute error.

    Returns:
        The Huber loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictions, "data", predictions))
        if targets is None:
            t = np.zeros_like(p)
        else:
            t = np.array(getattr(targets, "data", targets))
        diff = np.abs(p - t)
        return np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return predictions
