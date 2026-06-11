"""Regression losses."""

from typing import Any, Optional

from zero_jax import Array
from typing import cast
import ml_switcheroo.jnp as jnp


def squared_error(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the squared difference between predictions and targets.

    This loss is un-scaled and represents the squared Euclidean distance
    element-wise between the predicted values and the target values. If targets
    are omitted, they are assumed to be zero.

    Args:
        predictions: The predicted values as an Array.
        targets: The ground truth target labels. If None, targets are implicitly
            assumed to be an array of zeros with the same shape as predictions.

    Returns:
        An array of the same shape containing the element-wise squared errors.
    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    return cast(Array, (p - t) ** 2)


def l2_loss(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the standard L2 loss (half the squared error).

    The L2 loss is half of the squared error. The scaling factor of 0.5 is
    mathematically convenient because it cancels out when the derivative
    is computed with respect to the predictions.

    Args:
        predictions: The predicted values as an Array.
        targets: The ground truth target labels. If None, defaults to an array of zeros.

    Returns:
        An array of the same shape containing the element-wise L2 losses.
    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    return cast(Array, 0.5 * (p - t) ** 2)


def huber_loss(
    predictions: Array, targets: Optional[Array] = None, delta: float = 1.0
) -> Array:
    """Calculates the Huber loss, a robust loss function for regression.

    Huber loss transitions from a squared error for small errors (less than `delta`)
    to an absolute error for large errors. This makes it less sensitive to outliers
    than standard squared error.

    Args:
        predictions: The predicted values as an Array.
        targets: The ground truth target labels. If None, defaults to zeros.
        delta: The threshold parameter that determines where the loss transitions
            from quadratic to linear. Defaults to 1.0.

    Returns:
        An array of the same shape containing the element-wise Huber losses.
    """
    p = jnp.asarray(predictions)
    if targets is None:
        t = jnp.zeros_like(p)
    else:
        t = jnp.asarray(targets)
    diff = jnp.abs(p - t)
    return cast(
        Array, jnp.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    )
