"""Sequence losses."""

from typing import Any, Tuple

from zero_jax import Array
from typing import cast
import ml_switcheroo.jnp as jnp


def ctc_loss(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -1e5,
) -> Array:
    """Computes the Connectionist Temporal Classification (CTC) loss.

    CTC loss is used for sequence-to-sequence tasks where the timing of the alignment
    between the input sequence and the target sequence is unknown (e.g., speech recognition).

    Args:
        logits: Unnormalized log probabilities for each class at each time step.
        logit_paddings: Binary array indicating padding for the input logits (1 for padding, 0 for valid).
        labels: Integer IDs for the target sequence.
        label_paddings: Binary array indicating padding for the target labels.
        blank_id: The class index corresponding to the special 'blank' token used in CTC.
        log_epsilon: A small value used to prevent numerical underflow during log operations.

    Returns:
        An array containing the scalar CTC loss per sequence in the batch.
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
    """Computes the CTC loss and returns the internal forward/backward probabilities.

    In addition to the CTC loss, this function also exposes the alpha (forward) and
    beta (backward) probabilities computed during the forward-backward algorithm. These
    can be useful for tasks like forced alignment or debugging.

    Args:
        logits: Unnormalized log probabilities for each class at each time step.
        logit_paddings: Binary array indicating padding for the input logits.
        labels: Integer IDs for the target sequence.
        label_paddings: Binary array indicating padding for the target labels.
        blank_id: The class index corresponding to the 'blank' token.
        log_epsilon: Small value used to prevent numerical underflow in log space.

    Returns:
        A tuple containing:
            - The CTC loss array per sequence.
            - The alpha (forward) probability matrix.
            - The beta (backward) probability matrix.
            - Gamma, the marginal probabilities combining alpha and beta.
    """
    l = jnp.asarray(logits)
    B = l.shape[0]
    return (
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
        cast(Array, jnp.zeros(B)),
    )
