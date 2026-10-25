"""Ranking loss functions."""

from __future__ import annotations

import zero_chex

from zero_optax._tree import (
    asarray,
    log_softmax,
    reduce_max,
    reduce_mean,
    reduce_sum,
    zeros_like,
)
from zero_optax._tree import (
    where as where_select,
)
from zero_optax._types import ArrayLike, ReduceFn, Tensor


def _default_mean_reduce(x: Tensor) -> Tensor:
    """Reduce a tensor by computing its arithmetic mean across all elements.

    Args:
        x: Input tensor to reduce.

    Returns:
        Scalar mean Tensor.
    """
    return reduce_mean(x)


def ranking_softmax_loss(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    where: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduce_fn: ReduceFn | None = _default_mean_reduce,
) -> Tensor:
    """Compute the listwise ranking softmax cross-entropy loss.

    Evaluates `-sum_i labels_i * log(softmax(logits)_i)` along the last axis.

    Args:
        logits: Unnormalized item scores of shape `[..., list_size]`.
        labels: Non-negative relevance labels of shape `[..., list_size]`.
        where: Optional boolean mask of shape `[..., list_size]` indicating valid items.
        weights: Optional per-item or per-list weight tensor broadcastable to `labels`.
        reduce_fn: Optional reduction callable (defaults to arithmetic mean). If None,
            returns the unreduced per-list loss of shape `logits.shape[:-1]`.

    Returns:
        Ranking softmax loss Tensor, reduced if `reduce_fn` is provided.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    zero_chex.assert_equal_shape([l_arr, t_arr])
    zero_chex.assert_rank([l_arr, t_arr], [l_arr.ndim, l_arr.ndim])

    if where is not None:
        t_arr = where_select(asarray(where), t_arr, zeros_like(t_arr))

    if weights is not None:
        t_arr = t_arr * asarray(weights)

    log_probs = log_softmax(l_arr, axis=-1, where_mask=where)
    per_item = where_select(t_arr == 0.0, 0.0, t_arr * log_probs)
    loss = -reduce_sum(per_item, axis=-1)

    if reduce_fn is None:
        return loss
    if where is not None:
        valid_lists = reduce_max(where_select(asarray(where), 1.0, 0.0), axis=-1)
        valid_count = reduce_sum(valid_lists)
        if reduce_fn is _default_mean_reduce:
            return where_select(
                valid_count == 0.0,
                0.0,
                reduce_sum(loss * valid_lists)
                / where_select(valid_count == 0.0, 1.0, valid_count),
            )
    return reduce_fn(loss)


__all__ = [
    "ranking_softmax_loss",
]
