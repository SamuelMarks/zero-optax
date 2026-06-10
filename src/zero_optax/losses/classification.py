"""Classification losses."""

from typing import Any, Optional, Union, Tuple

import chex
import numpy as np


def hinge_loss(predictor_outputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Compute the hinge loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The hinge loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(targets, "data", targets))
        return np.maximum(0.0, 1.0 - p * t)
    return predictor_outputs


def perceptron_loss(predictor_outputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Compute the perceptron loss.

    Args:
        predictor_outputs: The output of the predictor.
        targets: The target labels.

    Returns:
        The perceptron loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(targets, "data", targets))
        return np.maximum(0.0, -p * t)
    return predictor_outputs


def safe_softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    axis: int = -1,
    where: Optional[chex.Array] = None,
) -> chex.Array:
    """Compute the safe softmax cross entropy.

    Args:
        logits: The logits.
        labels: The labels.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The safe softmax cross entropy loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        if where is not None:
            w = np.array(getattr(where, "data", where))
            l = np.where(w, l, -np.inf)
            t = np.where(w, t, 0.0)

        if l.size == 0:
            return np.zeros_like(t) if where is None else np.zeros_like(t)

        c = np.max(l, axis=axis, keepdims=True)
        c = np.where(np.isinf(c), 0.0, c)

        # Avoid nan from -inf - (-inf)
        l_minus_c = np.where(np.isinf(l), -np.inf, l - c)

        exp_l_c = np.exp(l_minus_c)
        if where is not None:
            exp_l_c = np.where(w, exp_l_c, 0.0)

        sumexp = np.sum(exp_l_c, axis=axis, keepdims=True)
        logsumexp = np.log(np.where(sumexp == 0, 1.0, sumexp))

        log_softmax = np.where(np.isinf(l), -np.inf, l_minus_c - logsumexp)
        if where is not None:
            log_softmax = np.where(w, log_softmax, 0.0)

        t_times_log = t * log_softmax
        t_times_log = np.where(t == 0.0, 0.0, t_times_log)
        res = -np.sum(t_times_log, axis=axis)
        return res
    return logits


def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    axis: int = -1,
    where: Optional[chex.Array] = None,
) -> chex.Array:
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
    logits: chex.Array,
    labels: chex.Array,
    axis: int = -1,
    where: Optional[chex.Array] = None,
) -> chex.Array:
    """Compute softmax cross entropy with integer labels.

    Args:
        logits: The logits.
        labels: The integer labels.
        axis: The axis along which to compute the softmax.
        where: A boolean array indicating where to compute the loss.

    Returns:
        The softmax cross entropy loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        if l.size == 0:
            return np.zeros_like(t) if where is None else np.zeros_like(t)

        c = np.max(l, axis=axis, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(l - c), axis=axis, keepdims=True))
        log_softmax = l - c - logsumexp
        res = -np.take_along_axis(
            log_softmax, np.expand_dims(t, axis), axis=axis
        ).squeeze(axis)
        if where is not None:
            w = np.array(getattr(where, "data", where))
            res = np.where(w, res, 0.0)
        return res
    return logits


def sigmoid_binary_cross_entropy(logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Compute the sigmoid binary cross entropy.

    Args:
        logits: The logits.
        labels: The labels.

    Returns:
        The sigmoid binary cross entropy loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        return np.maximum(l, 0) - l * t + np.log1p(np.exp(-np.abs(l)))
    return logits


def sigmoid_focal_loss(
    logits: chex.Array,
    labels: chex.Array,
    alpha: Optional[float] = None,
    gamma: float = 2.0,
) -> chex.Array:
    """Compute the sigmoid focal loss.

    Args:
        logits: The logits.
        labels: The labels.
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter.

    Returns:
        The sigmoid focal loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        p = 1 / (1 + np.exp(-l))
        ce = np.maximum(l, 0) - l * t + np.log1p(np.exp(-np.abs(l)))
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce * ((1 - p_t) ** gamma)
        if alpha is not None:
            alpha_t = alpha * t + (1 - alpha) * (1 - t)
            loss = alpha_t * loss
        return loss
    return logits


def multiclass_hinge_loss(
    predictor_outputs: chex.Array, labels: chex.Array
) -> chex.Array:
    """Compute the multiclass hinge loss.

    Args:
        predictor_outputs: The predictor outputs.
        labels: The labels.

    Returns:
        The multiclass hinge loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(labels, "data", labels))
        correct_p = np.take_along_axis(p, np.expand_dims(t, -1), axis=-1)
        margins = np.maximum(0.0, 1.0 - correct_p + p)
        np.put_along_axis(margins, np.expand_dims(t, -1), 0.0, axis=-1)
        return np.sum(margins, axis=-1)
    return predictor_outputs


def multiclass_perceptron_loss(
    predictor_outputs: chex.Array, labels: chex.Array
) -> chex.Array:
    """Compute the multiclass perceptron loss.

    Args:
        predictor_outputs: The predictor outputs.
        labels: The labels.

    Returns:
        The multiclass perceptron loss.

    """
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        p = np.array(getattr(predictor_outputs, "data", predictor_outputs))
        t = np.array(getattr(labels, "data", labels))
        correct_p = np.take_along_axis(p, np.expand_dims(t, -1), axis=-1)
        margins = np.maximum(0.0, -correct_p + p)
        np.put_along_axis(margins, np.expand_dims(t, -1), 0.0, axis=-1)
        return np.sum(margins, axis=-1)
    return predictor_outputs


def poly_loss_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    epsilon: float = 2.0,
    axis: int = -1,
    where: Optional[chex.Array] = None,
) -> chex.Array:
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
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        l = np.array(getattr(logits, "data", logits))
        t = np.array(getattr(labels, "data", labels))
        ce = safe_softmax_cross_entropy(l, t, axis=axis, where=where)
        if where is not None:
            w = np.array(getattr(where, "data", where))
            l = np.where(w, l, -np.inf)
            t = np.where(w, t, 0.0)

        if l.size == 0:
            return np.zeros_like(t) if where is None else np.zeros_like(t)

        c = np.max(l, axis=axis, keepdims=True)
        c = np.where(np.isinf(c), 0.0, c)

        # Avoid nan from -inf - (-inf)
        l_minus_c = np.where(np.isinf(l), -np.inf, l - c)

        exp_l_c = np.exp(l_minus_c)
        if where is not None:
            exp_l_c = np.where(w, exp_l_c, 0.0)

        sumexp = np.sum(exp_l_c, axis=axis, keepdims=True)
        logsumexp = np.log(np.where(sumexp == 0, 1.0, sumexp))

        p = np.exp(l - c - logsumexp)
        if where is not None:
            p = np.where(w, p, 0.0)

        pt = np.sum(p * t, axis=axis)

        one_minus_pt = np.sum(t * (1.0 - p), axis=axis)
        res = ce + epsilon * one_minus_pt
        return res
    return logits
