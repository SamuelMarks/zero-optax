"""Classification losses."""

import zero_jax.numpy as jnp
from typing import Optional, Union, Tuple
import zero_jax as jax
import numpy as np

Array = Union[jax.Array, np.ndarray, np.bool_, np.number]


def hinge_loss(predictor_outputs: Array, targets: Array) -> Array:
    """Computes the hinge loss for binary classification."""
    return jnp.maximum(0.0, 1.0 - predictor_outputs * targets)


def perceptron_loss(
    predictor_outputs: Union[Array, float, int], targets: Union[Array, float, int]
) -> Array:
    """Binary perceptron loss."""
    return jnp.maximum(0.0, -predictor_outputs * targets)  # type: ignore


def safe_softmax_cross_entropy(logits: Array, labels: Array) -> Array:
    """Computes the softmax cross entropy between sets of logits and labels."""
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits_max = jax.lax.stop_gradient(logits_max)
    shifted_logits = logits - logits_max
    logsumexp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    log_probs = shifted_logits - logsumexp
    return -jnp.sum(labels * log_probs, axis=-1)


def softmax_cross_entropy(
    logits: Array,
    labels: Array,
    axis: Union[int, Tuple[int, ...], None] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes the softmax cross entropy between sets of logits and labels."""
    logits_max = jnp.max(
        logits, axis=axis, keepdims=True, where=where, initial=-jnp.inf
    )
    logits_max = jax.lax.stop_gradient(logits_max)
    shifted_logits = logits - logits_max

    exp_logits = jnp.exp(shifted_logits)
    sum_exp_logits = jnp.sum(exp_logits, axis=axis, keepdims=True, where=where)
    logsumexp = jnp.log(sum_exp_logits)

    log_probs = shifted_logits - logsumexp
    loss = -labels * log_probs  # type: ignore
    if where is not None:
        loss = jnp.where(where, loss, 0.0)
    return jnp.sum(loss, axis=axis)


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    axis: Union[int, Tuple[int, ...], None] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes softmax cross entropy between the logits and integer labels."""
    labels_one_hot = jax.nn.one_hot(labels, logits.shape[axis])  # type: ignore
    return softmax_cross_entropy(logits, labels_one_hot, axis=axis, where=where)


def sigmoid_binary_cross_entropy(logits, labels):
    """Computes element-wise sigmoid cross entropy given logits and labels."""
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1.0 - labels) * log_not_p


def sigmoid_focal_loss(
    logits: Array, labels: Array, alpha: Optional[float] = None, gamma: float = 2.0
) -> Array:
    """Sigmoid focal loss."""
    p = jax.nn.sigmoid(logits)
    bce = sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    focal_weight = (1.0 - p_t) ** gamma
    loss = focal_weight * bce
    if alpha is not None:
        alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
        loss = alpha_t * loss
    return loss


def multiclass_hinge_loss(scores: Array, labels: Array) -> Array:
    """Multiclass hinge loss."""
    labels_one_hot = jax.nn.one_hot(labels, scores.shape[-1])
    correct_scores = jnp.sum(scores * labels_one_hot, axis=-1, keepdims=True)
    margin = jnp.maximum(0.0, 1.0 - correct_scores + scores)
    margin = jnp.where(labels_one_hot, 0.0, margin)
    return jnp.max(margin, axis=-1)


def multiclass_perceptron_loss(scores: Array, labels: Array) -> Array:
    """Multiclass perceptron loss."""
    labels_one_hot = jax.nn.one_hot(labels, scores.shape[-1])
    correct_scores = jnp.sum(scores * labels_one_hot, axis=-1, keepdims=True)
    margin = jnp.maximum(0.0, -correct_scores + scores)
    margin = jnp.where(labels_one_hot, 0.0, margin)
    return jnp.max(margin, axis=-1)


def poly_loss_cross_entropy(
    logits: Array,
    labels: Array,
    epsilon: float = 2.0,
    axis: Union[int, Tuple[int, ...], None] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes PolyLoss between logits and labels."""
    ce = softmax_cross_entropy(logits, labels, axis=axis, where=where)
    pt = jnp.sum(jax.nn.softmax(logits, axis=axis) * labels, axis=axis, where=where)
    poly1 = epsilon * (1.0 - pt)
    if where is not None:
        poly1 = jnp.where(where, poly1, 0.0)
    return ce + poly1
