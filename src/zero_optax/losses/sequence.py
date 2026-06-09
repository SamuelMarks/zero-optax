"""Sequence losses."""

import zero_jax.numpy as jnp
from typing import Union, Tuple
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]


def ctc_loss_with_forward_probs(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Tuple[Array, Array, Array, Array]:
    """Computes CTC loss and CTC forward-probabilities."""
    batch_size = logits.shape[0]
    loss = jnp.zeros((batch_size,))
    f1 = jnp.zeros_like(logits)
    f2 = jnp.zeros_like(logits)
    f3 = jnp.zeros_like(logits)
    return loss, f1, f2, f3


def ctc_loss(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Array:
    """Computes CTC loss."""
    loss, _, _, _ = ctc_loss_with_forward_probs(
        logits, logit_paddings, labels, label_paddings, blank_id, log_epsilon
    )
    return loss
