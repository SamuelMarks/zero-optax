"""Ranking loss functions."""

from __future__ import annotations

from zero_optax._tree import (
    INF,
    NEG_INF,
    abs_tensor,
    asarray,
    exp,
    log,
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

    if where is not None:
        w_mask = asarray(where)
        t_arr = where_select(w_mask, t_arr, zeros_like(t_arr))
        l_arr = where_select(w_mask, l_arr, NEG_INF)

    if weights is not None:
        t_arr = t_arr * asarray(weights)

    c = reduce_max(l_arr, axis=-1, keepdims=True)
    c_safe = where_select(abs_tensor(c) == INF, 0.0, c)
    l_shifted = where_select(abs_tensor(l_arr) == INF, NEG_INF, l_arr - c_safe)
    exp_l = exp(l_shifted)
    if where is not None:
        exp_l = where_select(asarray(where), exp_l, 0.0)
    sum_exp = reduce_sum(exp_l, axis=-1, keepdims=True)
    log_norm = log(where_select(sum_exp == 0.0, 1.0, sum_exp))
    log_softmax = where_select(abs_tensor(l_arr) == INF, NEG_INF, l_shifted - log_norm)

    per_item = where_select(t_arr == 0.0, 0.0, t_arr * log_softmax)
    if where is not None:
        per_item = where_select(asarray(where), per_item, 0.0)

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
