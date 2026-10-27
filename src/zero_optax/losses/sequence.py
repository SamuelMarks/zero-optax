"""Connectionist Temporal Classification (CTC) sequence loss functions."""

from __future__ import annotations

from zero_optax._tree import (
    NEG_INF,
    abs_tensor,
    arange,
    asarray,
    exp,
    expand_dims,
    log1p,
    log_softmax,
    maximum,
    minimum,
    ones,
    reduce_sum,
    reshape,
    squeeze,
    stack,
    take_along_axis,
    where,
    zeros,
)
from zero_optax._types import ArrayLike, Tensor


def _logaddexp(a: Tensor, b: Tensor) -> Tensor:
    """Compute element-wise numerically stable `log(exp(a) + exp(b))` on Tensors.

    Args:
        a: First log-probability Tensor.
        b: Second log-probability Tensor.

    Returns:
        Combined log-probability Tensor.
    """
    both_neg_inf = (a == NEG_INF) & (b == NEG_INF)
    m = maximum(a, b)
    m_safe = where(both_neg_inf, 0.0, m)
    diff = where(both_neg_inf, 0.0, -abs_tensor(a - b))
    res = m_safe + log1p(exp(diff))
    return where(both_neg_inf, NEG_INF, res)


def _slice_last_dim(x: Tensor, start: int, stop: int) -> Tensor:
    """Slice a 2D Tensor `x` along its last dimension `[start:stop]` differentiably.

    Args:
        x: 2D Tensor of shape `(B, D)`.
        start: Inclusive start index along axis `-1`.
        stop: Exclusive stop index along axis `-1`.

    Returns:
        Sliced 2D Tensor of shape `(B, stop - start)`.
    """
    idx = expand_dims(arange(start, stop), 0)
    return take_along_axis(x, idx, axis=-1)


def _concat_head_and_tail(head: Tensor, tail: Tensor, total_cols: int) -> Tensor:
    """Concatenate a 1-column `head` `(B, 1)` and `(total_cols - 1)`-column `tail` along axis `-1`.

    Args:
        head: Leftmost column Tensor of shape `(B, 1)`.
        tail: Remaining columns Tensor of shape `(B, total_cols - 1)`.
        total_cols: Total number of output columns (`>= 1`).

    Returns:
        Combined 2D Tensor of shape `(B, total_cols)`.
    """
    if total_cols <= 1:
        return head
    col_idx = expand_dims(arange(total_cols), 0)
    head_bcast = take_along_axis(head, zeros((1, total_cols), dtype=int), axis=-1)
    tail_idx = maximum(col_idx - 1, 0)
    tail_shifted = take_along_axis(tail, tail_idx, axis=-1)
    return where(col_idx == 0, head_bcast, tail_shifted)


def _select_time_slice(log_probs: Tensor, t: int, max_time: int) -> Tensor:
    """Extract time step `t` from 3D `log_probs` `(B, T, K)` differentiably.

    Args:
        log_probs: 3D Tensor of shape `(B, T, K)`.
        t: Time frame index in `[0, T)`.
        max_time: Total number of time frames `T`.

    Returns:
        2D Tensor of shape `(B, K)` for frame `t`.
    """
    t_mask = reshape(arange(max_time), (1, max_time, 1)) == t
    return reduce_sum(where(t_mask, log_probs, 0.0), axis=1)


def ctc_loss_with_forward_probs(
    logits: ArrayLike,
    logit_paddings: ArrayLike,
    labels: ArrayLike,
    label_paddings: ArrayLike,
    *,
    blank_id: int = 0,
    log_epsilon: float | ArrayLike = -100000.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute CTC loss and forward log-alpha blank and non-blank probabilities.

    Args:
        logits: 3D array of shape `(B, T, K)` containing unnormalized class log-probabilities.
        logit_paddings: 2D array of shape `(B, T)` with `1.0` indicating padded frames and `0.0` valid frames.
        labels: 2D integer array of shape `(B, N)` containing ground-truth label indices.
        label_paddings: 2D array of shape `(B, N)` with `1.0` indicating padded labels and `0.0` valid labels.
        blank_id: Class index reserved for the CTC blank token.
        log_epsilon: Numerically stable log-space approximation of `log(0)`.

    Returns:
        Tuple `(per_seq_loss, logalpha_blank, logalpha_nonblank)` where `per_seq_loss`
        has shape `(B,)` and `logalpha_blank` / `logalpha_nonblank` have shapes
        `(T, B, N + 1)` and `(T, B, N)` respectively.
    """
    logits_t = asarray(logits)
    logit_pad_t = asarray(logit_paddings)
    labels_t = asarray(labels)
    label_pad_t = asarray(label_paddings)
    log_eps_t = asarray(log_epsilon)

    batch_size, max_time, _ = logits_t.shape
    _, max_label_len = labels_t.shape

    log_probs = log_softmax(logits_t, axis=-1)

    n_plus_1_idx = expand_dims(arange(max_label_len + 1), 0)
    phi_prev = where(
        n_plus_1_idx == 0,
        0.0,
        ones((batch_size, max_label_len + 1)) * log_eps_t,
    )
    emit_prev = ones((batch_size, max_label_len)) * log_eps_t

    if max_label_len > 1:
        left_lbl = _slice_last_dim(labels_t, 0, max_label_len - 1)
        right_lbl = _slice_last_dim(labels_t, 1, max_label_len)
        same_adj = asarray(left_lbl == right_lbl)
        col_idx_n = expand_dims(arange(max_label_len), 0)
        adj_idx = minimum(col_idx_n, max_label_len - 2)
        same_padded = take_along_axis(where(same_adj, 1.0, 0.0), adj_idx, axis=-1)
        repeat_flags = where(col_idx_n < (max_label_len - 1), same_padded, 0.0)
    else:
        repeat_flags = ones((batch_size, max_label_len)) * 0.0

    phi_steps: list[Tensor] = []
    emit_steps: list[Tensor] = []

    for t in range(max_time):
        pad_t = _slice_last_dim(logit_pad_t, t, t + 1)
        lp_frame = _select_time_slice(log_probs, t, max_time)
        lp_blank = _slice_last_dim(lp_frame, blank_id, blank_id + 1)
        lp_emit = take_along_axis(lp_frame, labels_t, axis=-1)

        prev_phi_orig = phi_prev
        prev_phi_head = _slice_last_dim(prev_phi_orig, 0, 1)
        prev_phi_rest = _slice_last_dim(prev_phi_orig, 1, max_label_len + 1)
        prev_phi_tail = _logaddexp(
            prev_phi_rest,
            emit_prev + log_eps_t * repeat_flags,
        )
        if max_label_len > 1:
            prev_phi_prefix = _concat_head_and_tail(
                prev_phi_head,
                _slice_last_dim(prev_phi_tail, 0, max_label_len - 1),
                max_label_len,
            )
        else:
            prev_phi_prefix = prev_phi_head

        next_emit = _logaddexp(prev_phi_prefix + lp_emit, emit_prev + lp_emit)

        next_phi_0 = prev_phi_head + lp_blank
        next_phi_tail = _logaddexp(
            prev_phi_tail + lp_blank,
            emit_prev + lp_blank + log_eps_t * (1.0 - repeat_flags),
        )
        next_phi = _concat_head_and_tail(
            next_phi_0,
            next_phi_tail,
            max_label_len + 1,
        )

        emit_prev = pad_t * emit_prev + (1.0 - pad_t) * next_emit
        phi_prev = pad_t * prev_phi_orig + (1.0 - pad_t) * next_phi
        phi_steps.append(phi_prev)
        emit_steps.append(emit_prev)

    if max_time > 0:
        last_phi_0 = _slice_last_dim(phi_steps[-1], 0, 1)
        last_phi_tail = _logaddexp(
            _slice_last_dim(phi_steps[-1], 1, max_label_len + 1),
            emit_steps[-1],
        )
        last_phi = _concat_head_and_tail(
            last_phi_0,
            last_phi_tail,
            max_label_len + 1,
        )
        phi_steps[-1] = last_phi
        phi_out = stack(phi_steps, axis=0)
        emit_out = stack(emit_steps, axis=0)
    else:
        last_phi = phi_prev
        phi_out = ones((0, batch_size, max_label_len + 1)) * 0.0
        emit_out = ones((0, batch_size, max_label_len)) * 0.0

    valid_label_counts = reduce_sum(1.0 - label_pad_t, axis=-1, keepdims=True)
    target_indices = asarray(valid_label_counts, dtype=int)
    per_seq_loglike = squeeze(
        take_along_axis(last_phi, target_indices, axis=-1), axis=-1
    )
    per_seq_loss = -per_seq_loglike

    return (
        per_seq_loss,
        phi_out,
        emit_out,
    )


def ctc_loss(
    logits: ArrayLike,
    logit_paddings: ArrayLike,
    labels: ArrayLike,
    label_paddings: ArrayLike,
    *,
    blank_id: int = 0,
    log_epsilon: float | ArrayLike = -100000.0,
) -> Tensor:
    """Compute Connectionist Temporal Classification (CTC) loss per sequence.

    Args:
        logits: 3D array of shape `(B, T, K)` containing unnormalized class log-probabilities.
        logit_paddings: 2D array of shape `(B, T)` with `1.0` indicating padded frames.
        labels: 2D integer array of shape `(B, N)` containing ground-truth label indices.
        label_paddings: 2D array of shape `(B, N)` with `1.0` indicating padded labels.
        blank_id: Class index reserved for the CTC blank token.
        log_epsilon: Numerically stable log-space approximation of `log(0)`.

    Returns:
        1D Tensor of shape `(B,)` containing the CTC loss for each sequence in the batch.
    """
    per_seq_loss, _, _ = ctc_loss_with_forward_probs(
        logits,
        logit_paddings,
        labels,
        label_paddings,
        blank_id=blank_id,
        log_epsilon=log_epsilon,
    )
    return per_seq_loss


__all__ = [
    "ctc_loss",
    "ctc_loss_with_forward_probs",
]
