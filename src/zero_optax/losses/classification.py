"""Classification losses."""

from typing import Any, Optional, Union, Tuple

from zero_jax import Array
from typing import cast
import zero_jax.numpy as jnp
import zero_jax.nn as jnn


def hinge_loss(predictor_outputs: Array, targets: Array) -> Array:
    """Compute the hinge loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The hinge loss.

    """
    p = jnp.asarray(predictor_outputs)
    t = jnp.asarray(targets)
    return cast(Array, jnp.maximum(0.0, 1.0 - p * t))


def perceptron_loss(predictor_outputs: Array, targets: Array) -> Array:
    """Compute the perceptron loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The perceptron loss.

    """
    p = jnp.asarray(predictor_outputs)
    t = jnp.asarray(targets)
    return cast(Array, jnp.maximum(0.0, -p * t))


def safe_softmax_cross_entropy(
    logits: Array,
    labels: Array,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Compute the safe softmax cross entropy.

    Args:
        logits: The logits.
        labels: The labels.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The safe softmax cross entropy loss.

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
    """Compute the softmax cross entropy.

    Args:
        logits: The logits.
        labels: The labels.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The softmax cross entropy loss.

    """
    return safe_softmax_cross_entropy(logits, labels, axis=axis, where=where)


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Compute softmax cross entropy with integer labels.

    Args:
        logits: The logits.
        labels: The integer labels.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The softmax cross entropy loss.

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
    """Compute the sigmoid binary cross entropy.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The sigmoid binary cross entropy loss.

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
    """Compute the sigmoid focal loss.

    Args:
        logits: The logits.
        labels: The labels.
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter.

    Returns:
        The sigmoid focal loss.

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


def multiclass_hinge_loss(predictor_outputs: Array, labels: Array) -> Array:
    """Compute the multiclass hinge loss.

    Args:
        predictor_outputs: The predictor outputs.
        labels: The labels.

    Returns:
        The multiclass hinge loss.

    """
    p = jnp.asarray(predictor_outputs)
    t = jnp.asarray(labels)
    correct_p = jnp.take_along_axis(p, jnp.expand_dims(t, -1), axis=-1)
    margins = jnp.maximum(0.0, 1.0 - correct_p + p)
    mask = jnn.one_hot(t, p.shape[-1])
    margins = jnp.where(mask, 0.0, margins)
    return cast(Array, jnp.sum(margins, axis=-1))


def multiclass_perceptron_loss(predictor_outputs: Array, labels: Array) -> Array:
    """Compute the multiclass perceptron loss.

    Args:
        predictor_outputs: The predictor outputs.
        labels: The labels.

    Returns:
        The multiclass perceptron loss.

    """
    p = jnp.asarray(predictor_outputs)
    t = jnp.asarray(labels)
    correct_p = jnp.take_along_axis(p, jnp.expand_dims(t, -1), axis=-1)
    margins = jnp.maximum(0.0, -correct_p + p)
    mask = jnn.one_hot(t, p.shape[-1])
    margins = jnp.where(mask, 0.0, margins)
    return cast(Array, jnp.sum(margins, axis=-1))


def poly_loss_cross_entropy(
    logits: Array,
    labels: Array,
    epsilon: float = 2.0,
    axis: int = -1,
    where: Optional[Array] = None,
) -> Array:
    """Compute the poly loss cross entropy.

    Args:
        logits: The logits.
        labels: The labels.
        epsilon: The epsilon parameter.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The poly loss cross entropy loss.

    """
    l = jnp.asarray(logits)
    t = jnp.asarray(labels)
    ce = safe_softmax_cross_entropy(l, t, axis=axis, where=where)
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
