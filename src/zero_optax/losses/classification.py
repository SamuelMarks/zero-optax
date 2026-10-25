"""Classification and divergence loss functions."""

from __future__ import annotations

import zero_chex

from zero_optax._tree import (
    abs_tensor,
    arange,
    asarray,
    exp,
    expand_dims,
    log,
    log1p,
    log_softmax,
    maximum,
    reduce_max,
    reduce_sum,
    sigmoid,
    softmax,
    squeeze,
    take_along_axis,
    zeros,
)
from zero_optax._tree import (
    where as where_select,
)
from zero_optax._types import ArrayLike, Tensor


def hinge_loss(
    predictor_outputs: ArrayLike,
    targets: ArrayLike,
) -> Tensor:
    """Compute the binary hinge loss `max(0, 1 - predictor_outputs * targets)`.

    Args:
        predictor_outputs: Raw decision function outputs.
        targets: Binary target values in `{-1, +1}`.

    Returns:
        Element-wise hinge loss Tensor.
    """
    p = asarray(predictor_outputs)
    t = asarray(targets)
    return maximum(0.0, 1.0 - p * t)


def perceptron_loss(
    predictor_outputs: ArrayLike,
    targets: ArrayLike,
) -> Tensor:
    """Compute the binary perceptron loss `max(0, -predictor_outputs * targets)`.

    Args:
        predictor_outputs: Raw decision function scores.
        targets: Binary target values in `{-1, +1}`.

    Returns:
        Element-wise perceptron loss Tensor.
    """
    p = asarray(predictor_outputs)
    t = asarray(targets)
    return maximum(0.0, -p * t)


def _reduced_empty_shape(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Compute output shape when reducing an array with a zero-length dimension.

    Args:
        shape: Input tensor shape containing at least one zero dimension.
        axis: Axis or axes reduced over.

    Returns:
        Shape tuple after removing the reduced axes.
    """
    if axis is None:
        return ()
    ndim = len(shape)
    axes = (axis,) if isinstance(axis, int) else axis
    norm_axes = {a if a >= 0 else ndim + a for a in axes}
    return tuple(dim for idx, dim in enumerate(shape) if idx not in norm_axes)


def _safe_softmax_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute numerically stable softmax cross-entropy handling `-inf` logits and masks.

    Args:
        logits: Unnormalized log probabilities.
        labels: Target probability distribution along `axis`.
        axis: Class axis or axes along which the softmax is computed.
        where: Optional boolean mask indicating elements to include.

    Returns:
        Cross-entropy loss Tensor with `axis` reduced.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    if 0 in l_arr.shape:
        return zeros(_reduced_empty_shape(l_arr.shape, axis))

    if where is not None:
        t_arr = where_select(asarray(where), t_arr, 0.0)

    log_probs = log_softmax(l_arr, axis=axis, where_mask=where)
    t_times_log = where_select(t_arr == 0.0, 0.0, t_arr * log_probs)
    return -reduce_sum(t_times_log, axis=axis)


def safe_softmax_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute softmax cross-entropy treating `0 * log(0)` as `0` when `logits = -inf`.

    Args:
        logits: Unnormalized log probabilities of shape `[..., num_classes]`.
        labels: Valid probability distributions of shape broadcastable to `logits`.

    Returns:
        Cross-entropy loss Tensor of shape `[...]`.
    """
    return _safe_softmax_cross_entropy(logits, labels, axis=-1, where=None)


def softmax_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute standard categorical softmax cross-entropy between logits and labels.

    Args:
        logits: Unnormalized log probabilities for each class.
        labels: One-hot or probability target distribution across classes.
        axis: Axis or axes over which to compute the softmax.
        where: Optional boolean mask array to exclude masked elements.

    Returns:
        Calculated softmax cross-entropy loss Tensor.
    """
    return _safe_softmax_cross_entropy(logits, labels, axis=axis, where=where)


def softmax_cross_entropy_with_integer_labels(
    logits: ArrayLike,
    labels: ArrayLike,
    axis: int = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute softmax cross-entropy given mutually exclusive integer class labels.

    Args:
        logits: Unnormalized log probabilities for each class.
        labels: Integer indices representing the true class for each example.
        axis: The class axis over which to compute the softmax.
        where: Optional boolean mask array indicating which elements to include.

    Returns:
        Calculated softmax cross-entropy loss Tensor.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    if 0 in l_arr.shape:
        return zeros(t_arr.shape)

    log_probs = log_softmax(l_arr, axis=axis, where_mask=where)
    picked = take_along_axis(log_probs, expand_dims(t_arr, axis), axis=axis)
    return -squeeze(picked, axis=axis)


def sigmoid_binary_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute element-wise binary cross-entropy using a numerically stable formulation.

    Args:
        logits: Unnormalized log probabilities for the positive class.
        labels: Binary ground-truth labels in `{0, 1}` or probabilities in `[0, 1]`.

    Returns:
        Element-wise binary cross-entropy loss Tensor.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    return maximum(l_arr, 0.0) - l_arr * t_arr + log1p(exp(-abs_tensor(l_arr)))


def sigmoid_focal_loss(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    alpha: float | ArrayLike | None = None,
    gamma: float | ArrayLike = 2.0,
) -> Tensor:
    """Compute the sigmoid focal loss for class-imbalanced binary classification.

    Args:
        logits: Unnormalized log probabilities for each example.
        labels: Binary target labels in `{0, 1}`.
        alpha: Optional weighting factor in `(0, 1)` balancing positive vs negative examples.
        gamma: Focusing exponent that down-weights well-classified examples.

    Returns:
        Element-wise sigmoid focal loss Tensor.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    p = sigmoid(l_arr)
    ce = sigmoid_binary_cross_entropy(l_arr, t_arr)
    p_t = p * t_arr + (1.0 - p) * (1.0 - t_arr)
    loss = ce * ((1.0 - p_t) ** asarray(gamma))
    if alpha is not None:
        alpha_t = asarray(alpha) * t_arr + (1.0 - asarray(alpha)) * (1.0 - t_arr)
        loss = alpha_t * loss
    return loss


def multiclass_hinge_loss(
    scores: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute Crammer-Singer multiclass hinge loss for integer labels.

    Args:
        scores: Raw model scores for each class of shape `[..., num_classes]`.
        labels: Integer ground-truth class indices of shape `[...]`.

    Returns:
        Multiclass hinge loss per example of shape `[...]`.
    """
    p = asarray(scores)
    t = asarray(labels)
    zero_chex.assert_rank([p, t], [p.ndim, p.ndim - 1])
    correct_p = squeeze(take_along_axis(p, expand_dims(t, -1), axis=-1), axis=-1)
    one_hot = expand_dims(t, -1) == arange(p.shape[-1])
    margin_scores = p + where_select(one_hot, 0.0, 1.0)
    return reduce_max(margin_scores, axis=-1) - correct_p


def multiclass_perceptron_loss(
    scores: ArrayLike,
    labels: ArrayLike,
) -> Tensor:
    """Compute multiclass perceptron loss `max_j scores_j - scores_label`.

    Args:
        scores: Raw model scores for each class of shape `[..., num_classes]`.
        labels: Integer ground-truth class indices of shape `[...]`.

    Returns:
        Multiclass perceptron loss per example of shape `[...]`.
    """
    p = asarray(scores)
    t = asarray(labels)
    zero_chex.assert_rank([p, t], [p.ndim, p.ndim - 1])
    correct_p = squeeze(take_along_axis(p, expand_dims(t, -1), axis=-1), axis=-1)
    return reduce_max(p, axis=-1) - correct_p


def poly_loss_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    epsilon: float | ArrayLike = 2.0,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute PolyLoss-1 cross-entropy with polynomial expansion weight `epsilon`.

    Args:
        logits: Unnormalized log probabilities of shape `[..., num_classes]`.
        labels: Target probability distributions along `axis`.
        epsilon: Coefficient of the first polynomial expansion term `(1 - P_t)`.
        axis: Class axis or axes along which to compute softmax.
        where: Optional boolean mask array to exclude masked elements.

    Returns:
        PolyLoss cross-entropy Tensor with `axis` reduced.
    """
    l_arr = asarray(logits)
    t_arr = asarray(labels)
    if 0 in l_arr.shape:
        return zeros(_reduced_empty_shape(l_arr.shape, axis))

    ce = _safe_softmax_cross_entropy(l_arr, t_arr, axis=axis, where=where)
    if where is not None:
        t_arr = where_select(asarray(where), t_arr, 0.0)
    probs = softmax(l_arr, axis=axis, where_mask=where)
    one_minus_pt = reduce_sum(t_arr * (1.0 - probs), axis=axis)
    return ce + asarray(epsilon) * one_minus_pt


def kl_divergence(
    log_predictions: ArrayLike,
    targets: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute Kullback-Leibler divergence `sum(targets * (log(targets) - log_predictions))`.

    Args:
        log_predictions: Log-probabilities of predicted distribution.
        targets: Target probability distribution (non-negative).
        axis: Axis or axes along which to sum the divergence.
        where: Optional boolean mask specifying elements to include.

    Returns:
        KL divergence Tensor with `axis` reduced.
    """
    log_p = asarray(log_predictions)
    t_arr = asarray(targets)
    zero_chex.assert_equal_shape([log_p, t_arr])
    safe_t = where_select(t_arr <= 0.0, 1.0, t_arr)
    log_t = where_select(t_arr == 0.0, 0.0, log(safe_t))
    loss = where_select(t_arr == 0.0, 0.0, t_arr * (log_t - log_p))
    if where is not None:
        loss = where_select(asarray(where), loss, 0.0)
    return reduce_sum(loss, axis=axis)


def kl_divergence_with_log_targets(
    log_predictions: ArrayLike,
    log_targets: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute Kullback-Leibler divergence when both predictions and targets are in log-space.

    Args:
        log_predictions: Log-probabilities of predicted distribution.
        log_targets: Log-probabilities of target distribution.
        axis: Axis or axes along which to sum the divergence.
        where: Optional boolean mask specifying elements to include.

    Returns:
        KL divergence Tensor with `axis` reduced.
    """
    log_p = asarray(log_predictions)
    log_t = asarray(log_targets)
    zero_chex.assert_equal_shape([log_p, log_t])
    loss = exp(log_t) * (log_t - log_p)
    if where is not None:
        loss = where_select(asarray(where), loss, 0.0)
    return reduce_sum(loss, axis=axis)


def convex_kl_divergence(
    log_predictions: ArrayLike,
    targets: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
) -> Tensor:
    """Compute jointly convex Kullback-Leibler divergence `KL(targets || pred) + sum(pred - targets)`.

    Args:
        log_predictions: Log-probabilities of predicted distribution.
        targets: Target probability distribution.
        axis: Axis or axes along which to sum the divergence.
        where: Optional boolean mask specifying elements to include.

    Returns:
        Convex KL divergence Tensor with `axis` reduced.
    """
    log_p = asarray(log_predictions)
    t_arr = asarray(targets)
    zero_chex.assert_equal_shape([log_p, t_arr])
    kl = kl_divergence(log_p, t_arr, axis=axis, where=where)
    diff = exp(log_p) - t_arr
    if where is not None:
        diff = where_select(asarray(where), diff, 0.0)
    return kl + reduce_sum(diff, axis=axis)


def smooth_labels(
    labels: ArrayLike,
    alpha: float,
) -> Tensor:
    """Apply label smoothing to one-hot target distributions.

    Args:
        labels: One-hot encoded target labels of shape `[..., num_classes]`.
        alpha: Smoothing factor in `[0, 1]`.

    Returns:
        Smoothed label distribution `(1 - alpha) * labels + alpha / num_classes`.
    """
    t_arr = asarray(labels)
    num_categories = float(t_arr.shape[-1])
    return (1.0 - alpha) * t_arr + (alpha / num_categories)


__all__ = [
    "convex_kl_divergence",
    "hinge_loss",
    "kl_divergence",
    "kl_divergence_with_log_targets",
    "multiclass_hinge_loss",
    "multiclass_perceptron_loss",
    "perceptron_loss",
    "poly_loss_cross_entropy",
    "safe_softmax_cross_entropy",
    "sigmoid_binary_cross_entropy",
    "sigmoid_focal_loss",
    "smooth_labels",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
]
