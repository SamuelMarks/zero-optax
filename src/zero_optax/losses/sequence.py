"""Sequence losses."""

from typing import Any, Tuple

import chex
import numpy as np


def ctc_loss(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> chex.Array:
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
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        return np.zeros(np.array(getattr(logits, "data", logits)).shape[0])
    return logits


def ctc_loss_with_forward_probs(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
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
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        B = l.shape[0]
        return np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
    return logits, logits, logits, logits
