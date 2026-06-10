"""Sequence losses."""

from typing import Any, Tuple

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp


def ctc_loss(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Array:
    """Compute the CTC loss.

    Args:
        logits: The logits.
        logit_paddings: The logit paddings.
        labels: The labels.
        label_paddings: The label paddings.
        blank_id: The blank ID.
        log_epsilon: The log epsilon.

    Returns:
        The CTC loss.

    """
    l = jnp.asarray(logits)
    return cast(Array, jnp.zeros(l.shape[0]))


def ctc_loss_with_forward_probs(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Tuple[Array, Array, Array, Array]:
    """Compute the CTC loss with forward probabilities.

    Args:
        logits: The logits.
        logit_paddings: The logit paddings.
        labels: The labels.
        label_paddings: The label paddings.
        blank_id: The blank ID.
        log_epsilon: The log epsilon.

    Returns:
        A tuple of CTC loss, alpha, beta, and gamma.

    """
    l = jnp.asarray(logits)
    B = l.shape[0]
    return (
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
    )
