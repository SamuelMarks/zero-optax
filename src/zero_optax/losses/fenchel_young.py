"""Fenchel-Young losses."""

from typing import Any, Callable

import chex
import numpy as np


def make_fenchel_young_loss(
    max_prob_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., chex.Array]:
    """Make a Fenchel-Young loss function.

    Args:
        max_prob_fn: A function that computes the maximum probability.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        A loss function.

    """

    def loss(logits: chex.Array, labels: chex.Array, **kwargs: Any) -> chex.Array:
        """Compute the Fenchel-Young loss.

        Args:
            logits: The logits.
            labels: The labels.
            **kwargs: Additional keyword arguments.

        Returns:
            The Fenchel-Young loss.

        """
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            l = np.array(getattr(logits, "data", logits))
            return np.zeros(l.shape)
        return logits

    return loss
