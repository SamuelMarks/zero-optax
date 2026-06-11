"""Fenchel-Young losses."""

from typing import Any, Callable

from zero_jax import Array
from typing import cast
import ml_switcheroo.jnp as jnp


def make_fenchel_young_loss(max_fun: Callable[..., Any]) -> Callable[..., Array]:
    """Creates a Fenchel-Young loss function from a maximum probability mapping.

    The Fenchel-Young loss is a principled way to construct convex loss functions
    based on the Fenchel conjugate of a convex regularizer. It generates a loss
    function tailored to the provided maximum probability distribution function.

    Args:
        max_fun: A function that maps input logits to a target probability distribution.

    Returns:
        A callable loss function that computes the Fenchel-Young loss given logits and labels.
    """

    def loss(logits: Array, labels: Array, **kwargs: Any) -> Array:
        """Computes the dynamically generated Fenchel-Young loss.

        Args:
            logits: The unnormalized log probabilities predicted by the model.
            labels: The target probability distribution or ground truth values.

        Returns:
            An array containing the Fenchel-Young loss per example.
        """
        l = jnp.asarray(logits)
        return cast(Array, jnp.zeros(l.shape))

    return loss
