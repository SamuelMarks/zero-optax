"""Classification losses."""

from typing import Any, Optional, Union, Tuple

import chex
from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp
import zero_jax.nn as jnn


def hinge_loss(scores: chex.Array, targets: chex.Array) -> chex.Array:
    """Compute the hinge loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The hinge loss.

    """
    p = jnp.asarray(scores)
    t = jnp.asarray(targets)
    return cast(Array, jnp.maximum(0.0, 1.0 - p * t))


def perceptron_loss(scores: chex.Numeric, targets: chex.Numeric) -> chex.Numeric:
    """Compute the perceptron loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The perceptron loss.

    """
    p = jnp.asarray(scores)
    t = jnp.asarray(targets)
    return cast(Array, jnp.maximum(0.0, -p * t))


def _safe_softmax_cross_entropy(
    logits: Array, labels: Array, axis: int = -1, where: Optional[Array] = None
) -> Array:
    """Computes the softmax cross entropy loss safely, avoiding NaN values.

    This function calculates the cross entropy between a probability distribution
    (from the softmax of logits) and a target distribution (labels). It is designed
    to be numerically stable, handling `-inf` logits correctly.

    Args:
        logits: Unnormalized log probabilities.
        labels: A valid probability distribution (non-negative, sum to 1).
        axis: The class axis along which the softmax is computed. Defaults to -1.
        where: An optional boolean mask array indicating which elements to include
            in the computation. Elements where `where` is False are ignored.

    Returns:
        An array containing the cross entropy loss per example.
    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    if where is not None:
        w = jnp.asarray(where)
        l = jnp.where(w, l, -jnp.inf)
        t = jnp.where(w, t, 0.0)

    if 0 in l.shape:
        return (
            cast(Array, jnp.zeros_like(t))
            if where is None
            else cast(Array, jnp.zeros_like(t))
        )

    c = jnp.max(l, axis=axis, keepdims=True)
    c = jnp.where((jnp.abs(c) == jnp.inf), 0.0, c)

    # Avoid nan from -inf - (-inf)
    l_minus_c = jnp.where((jnp.abs(l) == jnp.inf), -jnp.inf, l - c)

    exp_l_c = jnp.exp(l_minus_c)
    if where is not None:
        exp_l_c = jnp.where(w, exp_l_c, 0.0)

    sumexp = jnp.sum(exp_l_c, axis=axis, keepdims=True)
    logsumexp = jnp.log(jnp.where(sumexp == 0, 1.0, sumexp))

    log_softmax = jnp.where((jnp.abs(l) == jnp.inf), -jnp.inf, l_minus_c - logsumexp)
    if where is not None:
        log_softmax = jnp.where(w, log_softmax, 0.0)

    t_times_log = t * log_softmax
    t_times_log = jnp.where(t == 0.0, 0.0, t_times_log)
    res = -jnp.sum(t_times_log, axis=axis)
    return cast(Array, res)


def softmax_cross_entropy(
    logits: Array,
    labels: Array,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes the standard softmax cross entropy loss.

    This function wraps `safe_softmax_cross_entropy` and calculates the cross
    entropy between predicted probabilities (derived from logits) and target
    labels.

    Args:
        logits: Unnormalized log probabilities for each class.
        labels: Target probability distribution across classes.
        axis: The axis over which to compute the softmax. Defaults to -1.
        where: Optional boolean array to mask certain inputs from the loss computation.

    Returns:
        An array containing the calculated softmax cross entropy loss.
    """
    return _safe_softmax_cross_entropy(logits, labels, axis=axis, where=where)


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes softmax cross entropy given mutually exclusive integer labels.

    This function avoids the need to explicitly one-hot encode the labels. It
    calculates the cross-entropy loss by utilizing the integer class indices directly.

    Args:
        logits: Unnormalized log probabilities for each class.
        labels: Integer indices representing the true class for each example.
        axis: The class axis over which to compute the softmax. Defaults to -1.
        where: Optional boolean mask array indicating which elements to compute.

    Returns:
        An array containing the calculated softmax cross entropy loss.
    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    if 0 in l.shape:
        return (
            cast(Array, jnp.zeros_like(t))
            if where is None
            else cast(Array, jnp.zeros_like(t))
        )

    c = jnp.max(l, axis=axis, keepdims=True)
    logsumexp = jnp.log(jnp.sum(jnp.exp(l - c), axis=axis, keepdims=True))
    log_softmax = l - c - logsumexp
    res = -jnp.squeeze(
        jnp.take_along_axis(log_softmax, jnp.expand_dims(t, axis), axis=axis), axis=axis
    )
    if where is not None:
        w = jnp.asarray(where)
        res = jnp.where(w, res, 0.0)
    return cast(Array, res)


def sigmoid_binary_cross_entropy(logits: Array, labels: Array) -> Array:
    """Computes the binary cross entropy loss utilizing a numerically stable sigmoid.

    This function combines a sigmoid activation and a binary cross entropy loss
    into a single step for better numerical stability when extreme logits are present.

    Args:
        logits: Unnormalized log probabilities for the positive class.
        labels: The ground truth binary labels (typically 0 or 1).

    Returns:
        An array containing the element-wise binary cross entropy loss.
    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    return cast(Array, jnp.maximum(l, 0) - l * t + jnp.log1p(jnp.exp(-jnp.abs(l))))


def sigmoid_focal_loss(
    logits: Array,
    labels: Array,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
) -> Array:
    """Computes the sigmoid focal loss for dense object detection or classification.

    Focal loss dynamically scales the cross entropy loss based on confidence, down-weighting
    well-classified examples. This helps prevent the vast number of easy negatives from
    overwhelming the detector during training.

    Args:
        logits: Unnormalized log probabilities.
        labels: The ground truth binary labels (0 or 1).
        alpha: Optional weighting factor for the positive class to handle class imbalance.
        gamma: Focusing parameter that smoothly adjusts the rate at which easy
            examples are down-weighted. Higher values increase the focusing effect.

    Returns:
        An array containing the computed sigmoid focal loss.
    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    p = jnn.sigmoid(l)
    ce = jnp.maximum(l, 0) - l * t + jnp.log1p(jnp.exp(-jnp.abs(l)))
    p_t = p * t + (1 - p) * (1 - t)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * t + (1 - alpha) * (1 - t)
        loss = alpha_t * loss
    return cast(Array, loss)


def multiclass_hinge_loss(scores: Array, labels: Array) -> Array:
    """Computes the multiclass formulation of the hinge loss.

    This loss penalizes predictions where the correct class's score is not greater
    than any incorrect class's score by at least a margin of 1.0. It is useful
    for training multiclass support vector machines.

    Args:
        scores: The raw scores or logits predicted for each class.
        labels: Integer indices representing the true class for each example.

    Returns:
        An array containing the multiclass hinge loss for each example.
    """
    p = jnp.asarray(scores)
    t = jnp.asarray(labels)
    correct_p = jnp.take_along_axis(p, jnp.expand_dims(t, -1), axis=-1)
    margins = jnp.maximum(0.0, 1.0 - correct_p + p)
    mask = jnp.expand_dims(t, -1) == jnp.arange(p.shape[-1])
    margins = jnp.where(mask, 0.0, margins)
    return cast(Array, jnp.sum(margins, axis=-1))


def multiclass_perceptron_loss(scores: Array, labels: Array) -> Array:
    """Computes the multiclass perceptron loss.

    Similar to multiclass hinge loss, but without a margin. It penalizes predictions
    only when an incorrect class has a higher score than the true class.

    Args:
        scores: The raw scores or logits predicted for each class.
        labels: Integer indices representing the true class for each example.

    Returns:
        An array containing the multiclass perceptron loss for each example.
    """
    p = jnp.asarray(scores)
    t = jnp.asarray(labels)
    correct_p = jnp.take_along_axis(p, jnp.expand_dims(t, -1), axis=-1)
    margins = jnp.maximum(0.0, -correct_p + p)
    mask = jnp.expand_dims(t, -1) == jnp.arange(p.shape[-1])
    margins = jnp.where(mask, 0.0, margins)
    return cast(Array, jnp.sum(margins, axis=-1))


def poly_loss_cross_entropy(
    logits: Array,
    labels: Array,
    epsilon: float = 2.0,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes the PolyLoss modification of cross-entropy.

    PolyLoss is an alternative to cross-entropy that adjusts the loss by adding
    a polynomial expansion term based on the leading polynomial coefficient. It
    often improves convergence and accuracy for classification tasks.

    Args:
        logits: Unnormalized log probabilities for each class.
        labels: A valid target probability distribution (non-negative, sum to 1).
        epsilon: The polynomial weighting coefficient adjusting the contribution
            of the leading polynomial term. Defaults to 2.0.
        axis: The class axis along which to compute the softmax. Defaults to -1.
        where: Optional boolean array to mask certain inputs from the loss computation.

    Returns:
        An array containing the calculated poly-loss cross entropy.
    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    ce = _safe_softmax_cross_entropy(l, t, axis=axis, where=where)
    if where is not None:
        w = jnp.asarray(where)
        l = jnp.where(w, l, -jnp.inf)
        t = jnp.where(w, t, 0.0)

    if 0 in l.shape:
        return (
            cast(Array, jnp.zeros_like(t))
            if where is None
            else cast(Array, jnp.zeros_like(t))
        )

    c = jnp.max(l, axis=axis, keepdims=True)
    c = jnp.where((jnp.abs(c) == jnp.inf), 0.0, c)

    # Avoid nan from -inf - (-inf)
    l_minus_c = jnp.where((jnp.abs(l) == jnp.inf), -jnp.inf, l - c)

    exp_l_c = jnp.exp(l_minus_c)
    if where is not None:
        exp_l_c = jnp.where(w, exp_l_c, 0.0)

    sumexp = jnp.sum(exp_l_c, axis=axis, keepdims=True)
    logsumexp = jnp.log(jnp.where(sumexp == 0, 1.0, sumexp))

    p = jnp.exp(l - c - logsumexp)
    if where is not None:
        p = jnp.where(w, p, 0.0)

    pt = jnp.sum(p * t, axis=axis)

    one_minus_pt = jnp.sum(t * (1.0 - p), axis=axis)
    res = ce + epsilon * one_minus_pt
    return cast(Array, res)


def safe_softmax_cross_entropy(logits: Array, labels: Array) -> Array:
    """Computes the safe softmax cross entropy."""
    return cast(
        Array, _safe_softmax_cross_entropy(cast(Array, logits), cast(Array, labels))
    )
