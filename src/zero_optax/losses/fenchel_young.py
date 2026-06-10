"""Fenchel-Young losses."""

from typing import Any, Callable

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp


def make_fenchel_young_loss(
    max_prob_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., Array]:
    """Make a Fenchel-Young loss function.

    Args:
        max_prob_fn: A function that computes the maximum probability.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        A loss function.

    """

    def loss(logits: Array, labels: Array, **kwargs: Any) -> Array:
        """Compute the Fenchel-Young loss.

        Args:
            logits: The logits.
            labels: The labels.
            **kwargs: Additional keyword arguments.

        Returns:
            The Fenchel-Young loss.

        """
        l = jnp.asarray(logits)
        return cast(Array, jnp.zeros(l.shape))

    return loss
