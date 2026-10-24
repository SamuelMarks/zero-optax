"""Metric learning and contrastive loss functions."""

from __future__ import annotations

from zero_optax._tree import (
    NEG_INF,
    abs_tensor,
    arange,
    asarray,
    expand_dims,
    logsumexp,
    maximum,
    reduce_sum,
    reshape,
    safe_norm,
    where,
)
from zero_optax._types import ArrayLike, Tensor


def triplet_margin_loss(
    anchors: ArrayLike,
    positives: ArrayLike,
    negatives: ArrayLike,
    axis: int = -1,
    norm_degree: float | int = 2,
    margin: float | ArrayLike = 1.0,
    eps: float | ArrayLike = 1e-6,
) -> Tensor:
    """Compute the triplet margin loss for metric learning.

    Evaluates `max(0, d(anchors, positives) - d(anchors, negatives) + margin)`
    where `d(x, y) = (sum(|x - y|^p) + eps)^(1/p)`.

    Args:
        anchors: Anchor embeddings tensor.
        positives: Positive embeddings tensor matching `anchors` shape.
        negatives: Negative embeddings tensor matching `anchors` shape.
        axis: Embedding feature dimension along which distances are computed.
        norm_degree: Degree `p` of the `L_p` distance norm.
        margin: Minimum desired distance margin between positive and negative pairs.
        eps: Small constant added inside the norm power to ensure differentiability at zero.

    Returns:
        Triplet margin loss Tensor with `axis` reduced.
    """
    a = asarray(anchors)
    p = asarray(positives)
    n = asarray(negatives)
    deg = float(norm_degree)
    inv_deg = 1.0 / deg
    eps_t = asarray(eps)
    margin_t = asarray(margin)

    pos_diff = abs_tensor(a - p) ** deg
    neg_diff = abs_tensor(a - n) ** deg

    pos_dist = (reduce_sum(pos_diff, axis=axis) + eps_t) ** inv_deg
    neg_dist = (reduce_sum(neg_diff, axis=axis) + eps_t) ** inv_deg

    return maximum(0.0, pos_dist - neg_dist + margin_t)


def ntxent(
    embeddings: ArrayLike,
    labels: ArrayLike,
    temperature: float | ArrayLike = 0.07,
) -> Tensor:
    """Compute the Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss.

    Args:
        embeddings: 2D batch of embeddings of shape `[batch, feature_length]`.
        labels: 1D integer group labels of shape `[batch]` identifying positive pairs.
        temperature: Positive temperature scaling parameter.

    Returns:
        Scalar Tensor holding the mean NT-Xent loss over all positive pairs.

    Raises:
        ValueError: If `embeddings` and `labels` have mismatched leading dimensions.
    """
    emb = asarray(embeddings)
    lbl = asarray(labels)
    if lbl.shape[0] != emb.shape[0]:
        raise ValueError(
            "Labels and embeddings must have the same leading dimension, found "
            f"{lbl.shape[0]} for labels and {emb.shape[0]} for embeddings."
        )
    batch_size = emb.shape[0]
    feat_dim = emb.shape[1] if emb.ndim > 1 else 1
    emb_2d = reshape(emb, (batch_size, feat_dim))
    lbl_1d = reshape(lbl, (batch_size,))
    temp_t = asarray(temperature)

    normed = emb_2d / safe_norm(emb_2d, 1e-6, axis=-1, keepdims=True)
    sims = reduce_sum(expand_dims(normed, 1) * expand_dims(normed, 0), axis=-1) / temp_t

    idx = arange(batch_size)
    eye_mask = expand_dims(idx, 1) == expand_dims(idx, 0)
    masked_sims = where(eye_mask, NEG_INF, sims)
    log_denom = logsumexp(masked_sims, axis=-1, keepdims=True)

    same_label = expand_dims(lbl_1d, 1) == expand_dims(lbl_1d, 0)
    pos_mask = where(eye_mask, False, same_label)
    pos_weights = where(pos_mask, 1.0, 0.0)
    match_count = reduce_sum(pos_weights)

    pair_losses = where(pos_mask, -(sims - log_denom), 0.0)
    total_loss = reduce_sum(pair_losses)
    safe_count = where(match_count == 0.0, 1.0, match_count)
    return where(match_count == 0.0, 0.0, total_loss / safe_count)


__all__ = [
    "ntxent",
    "triplet_margin_loss",
]
