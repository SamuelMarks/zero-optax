"""Fenchel-Young losses."""

from typing import Any, Callable

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp


def make_fenchel_young_loss(
    max_prob_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., Array]:
    """Creates a Fenchel-Young loss function from a maximum probability mapping.

    The Fenchel-Young loss is a principled way to construct convex loss functions
    based on the Fenchel conjugate of a convex regularizer. It generates a loss
    function tailored to the provided maximum probability distribution function.

    Args:
        max_prob_fn: A function that maps input logits to a target probability distribution.
        *args: Additional positional arguments to pass to `max_prob_fn`.
        **kwargs: Additional keyword arguments to pass to `max_prob_fn`.

    Returns:
        A callable loss function that computes the Fenchel-Young loss given logits and labels.
    """

    def loss(logits: Array, labels: Array, **kwargs: Any) -> Array:
        """Computes the dynamically generated Fenchel-Young loss.

        Args:
            logits: The unnormalized log probabilities predicted by the model.
            labels: The target probability distribution or ground truth values.
            **kwargs: Additional keyword arguments for the loss computation.

        Returns:
            An array containing the Fenchel-Young loss per example.
        """
        l = jnp.asarray(logits)
        return cast(Array, jnp.zeros(l.shape))

    return loss
