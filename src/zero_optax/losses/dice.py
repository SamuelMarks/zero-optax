"""Sørensen-Dice and Tversky segmentation loss functions."""

from __future__ import annotations

from zero_optax._tree import (
    asarray,
    exp,
    reduce_max,
    reduce_mean,
    reduce_sum,
    sigmoid,
    where,
)
from zero_optax._types import ArrayLike, Tensor


def _softmax_last_axis(logits: Tensor) -> Tensor:
    """Compute numerically stable softmax along the last axis (`-1`).

    Args:
        logits: Unnormalized log probabilities Tensor.

    Returns:
        Normalized probability distribution Tensor along axis `-1`.
    """
    c = reduce_max(logits, axis=-1, keepdims=True)
    exp_shifted = exp(logits - c)
    denom = reduce_sum(exp_shifted, axis=-1, keepdims=True)
    return exp_shifted / where(denom == 0.0, 1.0, denom)


def binary_dice_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    smooth: float | ArrayLike = 1.0,
    apply_sigmoid: bool = True,
) -> Tensor:
    """Compute the binary Sørensen-Dice loss.

    Args:
        predictions: Model predictions (logits if `apply_sigmoid=True`, probabilities
            otherwise) of shape `[..., num_features]`.
        targets: Binary target masks of shape matching `predictions`.
        smooth: Additive smoothing constant in numerator and denominator.
        apply_sigmoid: Whether to apply logistic sigmoid to `predictions` first.

    Returns:
        Binary Dice loss Tensor reduced along the last axis.
    """
    preds = asarray(predictions)
    targs = asarray(targets)
    probs = sigmoid(preds) if apply_sigmoid else preds
    smooth_t = asarray(smooth)
    intersection = reduce_sum(probs * targs, axis=-1)
    pred_sum = reduce_sum(probs, axis=-1)
    targ_sum = reduce_sum(targs, axis=-1)
    return 1.0 - (2.0 * intersection + smooth_t) / (pred_sum + targ_sum + smooth_t)


def dice_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    class_weights: ArrayLike | None = None,
    smooth: float | ArrayLike = 1.0,
    alpha: float | ArrayLike = 0.5,
    beta: float | ArrayLike = 0.5,
    apply_softmax: bool = True,
    reduction: str = "mean",
    ignore_background: bool = False,
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    """Compute multiclass Dice / Tversky loss for segmentation tasks.

    Args:
        predictions: Model predictions of shape `[..., spatial..., num_classes]`.
        targets: One-hot or soft target probabilities of shape matching `predictions`.
        class_weights: Optional per-class weight vector of shape `[num_classes]`.
        smooth: Laplace smoothing factor added to numerator and denominator.
        alpha: Weight applied to false positives (`0.5` for standard Dice).
        beta: Weight applied to false negatives (`0.5` for standard Dice).
        apply_softmax: Whether to apply softmax along the class axis (`-1`).
        reduction: Class reduction mode: `'mean'`, `'sum'`, or `'none'`.
        ignore_background: If True, exclude channel index `0` from the loss calculation.
        axis: Spatial axis or tuple of axes to sum over before class reduction.

    Returns:
        Computed Dice / Tversky loss Tensor.

    Raises:
        ValueError: If `reduction` is not one of `'mean'`, `'sum'`, or `'none'`.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"Unsupported reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
        )
    preds = asarray(predictions)
    targs = asarray(targets)
    probs = _softmax_last_axis(preds) if apply_softmax else preds

    if ignore_background and probs.shape[-1] > 1:
        probs = probs[..., 1:]
        targs = targs[..., 1:]
        if class_weights is not None:
            cw_full = asarray(class_weights)
            if cw_full.shape[-1] > probs.shape[-1]:
                class_weights = cw_full[..., 1:]

    tp = probs * targs
    fp = probs * (1.0 - targs)
    fn = (1.0 - probs) * targs

    if axis is not None:
        tp = reduce_sum(tp, axis=axis)
        fp = reduce_sum(fp, axis=axis)
        fn = reduce_sum(fn, axis=axis)
    elif probs.ndim > 2:
        spatial_axes = tuple(range(1, probs.ndim - 1))
        tp = reduce_sum(tp, axis=spatial_axes)
        fp = reduce_sum(fp, axis=spatial_axes)
        fn = reduce_sum(fn, axis=spatial_axes)

    smooth_t = asarray(smooth)
    alpha_t = asarray(alpha)
    beta_t = asarray(beta)

    numer = tp + smooth_t
    denom = tp + alpha_t * fp + beta_t * fn + smooth_t
    per_class_loss = 1.0 - (numer / denom)

    if class_weights is not None:
        cw = asarray(class_weights)
        per_class_loss = per_class_loss * cw

    if reduction == "none":
        return per_class_loss
    if reduction == "sum":
        return reduce_sum(per_class_loss, axis=-1)
    if class_weights is not None:
        cw_sum = reduce_sum(asarray(class_weights), axis=-1)
        return reduce_sum(per_class_loss, axis=-1) / where(cw_sum == 0.0, 1.0, cw_sum)
    return reduce_mean(per_class_loss, axis=-1)


def multiclass_generalized_dice_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    smooth: float | ArrayLike = 1.0,
    apply_softmax: bool = True,
    ignore_background: bool = False,
) -> Tensor:
    """Compute Multiclass Generalized Dice Loss weighted by inverse squared target volume.

    Args:
        predictions: Model predictions of shape `[batch, ..., num_classes]`.
        targets: Target probabilities of shape `[batch, ..., num_classes]`.
        smooth: Smoothing constant added to numerator and denominator.
        apply_softmax: Whether to apply softmax to `predictions` along axis `-1`.
        ignore_background: If True, drop channel `0` before computing the loss.

    Returns:
        Generalized Dice loss Tensor of shape `[batch]` (or scalar for 1D inputs).
    """
    preds = asarray(predictions)
    targs = asarray(targets)
    probs = _softmax_last_axis(preds) if apply_softmax else preds

    if ignore_background and probs.shape[-1] > 1:
        probs = probs[..., 1:]
        targs = targs[..., 1:]

    if probs.ndim > 2:
        spatial_axes: int | tuple[int, ...] = tuple(range(1, probs.ndim - 1))
        intersection = reduce_sum(probs * targs, axis=spatial_axes)
        pred_sum = reduce_sum(probs, axis=spatial_axes)
        targ_sum = reduce_sum(targs, axis=spatial_axes)
    else:
        intersection = probs * targs
        pred_sum = probs
        targ_sum = targs

    weights = 1.0 / (targ_sum * targ_sum + 1e-8)
    smooth_t = asarray(smooth)
    numer = 2.0 * reduce_sum(weights * intersection, axis=-1) + smooth_t
    denom = reduce_sum(weights * (pred_sum + targ_sum), axis=-1) + smooth_t
    return 1.0 - (numer / denom)


__all__ = [
    "binary_dice_loss",
    "dice_loss",
    "multiclass_generalized_dice_loss",
]
