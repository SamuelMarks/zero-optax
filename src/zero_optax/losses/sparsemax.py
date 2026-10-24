"""Sparsemax loss functions for binary and multiclass classification."""

from __future__ import annotations

from zero_optax._tree import (
    arange,
    asarray,
    expand_dims,
    maximum,
    reduce_sum,
    reshape,
    take_along_axis,
    where,
)
from zero_optax._types import ArrayLike, Tensor


def _sparse_plus(z: Tensor) -> Tensor:
    """Compute the binary sparsemax cumulant (`jax.nn.sparse_plus`).

    Defined piecewise as:
    - `0` for `z <= -1`
    - `0.25 * (z + 1)^2` for `-1 < z < 1`
    - `z` for `z >= 1`

    Args:
        z: Input score Tensor.

    Returns:
        Element-wise sparse_plus Tensor.
    """
    quad = 0.25 * (z + 1.0) * (z + 1.0)
    upper = where(z >= 1.0, z, quad)
    return where(z <= -1.0, 0.0, upper)


def sparsemax_loss(
    logits: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute the binary sparsemax loss `sparse_plus(where(labels, -logits, logits))`.

    Args:
        logits: Unnormalized binary score predictions.
        labels: Binary ground-truth target labels (`0` or `1`).

    Returns:
        Element-wise binary sparsemax loss Tensor matching the shape of `logits`.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    signed = where(t_arr > 0, -l_arr, l_arr)
    return _sparse_plus(signed)


def _simplex_support_mask_2d(scores_2d: Tensor) -> Tensor:
    """Determine the boolean active support mask of the simplex projection for each 2D row.

    Args:
        scores_2d: 2D Tensor of shape `(num_examples, num_classes)`.

    Returns:
        2D boolean Tensor of shape `(num_examples, num_classes)` with `1.0` for active
        support classes (`s_c > tau`) and `0.0` otherwise.
    """
    num_examples, num_classes = scores_2d.shape
    mask_rows: list[list[float]] = []
    for i in range(num_examples):
        row = [float(scores_2d[i, c].item()) for c in range(num_classes)]
        sorted_vals = sorted(row, reverse=True)
        cumsum = 0.0
        tau = 0.0
        for idx, val in enumerate(sorted_vals):
            k = idx + 1
            cumsum += val
            candidate_tau = (cumsum - 1.0) / float(k)
            if val > candidate_tau:
                tau = candidate_tau
        mask_rows.append([1.0 if v > tau else 0.0 for v in row])
    return asarray(mask_rows)


def multiclass_sparsemax_loss(
    scores: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute multiclass sparsemax loss for integer class labels.

    Args:
        scores: Unnormalized class scores of shape `[..., num_classes]`.
        labels: Integer indices of the true class of shape `[...]`.

    Returns:
        Multiclass sparsemax loss Tensor of shape `scores.shape[:-1]`.
    """
    s_arr = asarray(scores)
    l_arr = asarray(labels)
    lead_shape = s_arr.shape[:-1]
    num_classes = s_arr.shape[-1]

    num_examples = 1
    for dim in lead_shape:
        num_examples *= dim

    s_2d = reshape(s_arr, (num_examples, num_classes))
    l_1d = reshape(l_arr, (num_examples, 1))

    support_mask = _simplex_support_mask_2d(s_2d)
    support_count = reduce_sum(support_mask, axis=-1, keepdims=True)
    support_sum = reduce_sum(s_2d * support_mask, axis=-1, keepdims=True)
    tau = (support_sum - 1.0) / support_count
    proba = maximum(s_2d - tau, 0.0) * support_mask

    label_scores = take_along_axis(s_2d, l_1d, axis=-1)
    class_idx = expand_dims(arange(num_classes), 0)
    is_label = class_idx == l_1d
    shifted = where(is_label, 0.0, s_2d - label_scores)

    dot_term = reduce_sum(where(proba > 0.0, proba * shifted, 0.0), axis=-1)
    gini_term = 0.5 * (1.0 - reduce_sum(proba * proba, axis=-1))
    losses = dot_term + gini_term
    return reshape(losses, lead_shape)


__all__ = [
    "multiclass_sparsemax_loss",
    "sparsemax_loss",
]
