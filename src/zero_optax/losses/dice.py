"""Sørensen-Dice and Tversky segmentation loss functions."""

from __future__ import annotations

from zero_optax._tree import (
    asarray,
    expand_dims,
    reduce_mean,
    reduce_sum,
    sigmoid,
    softmax,
    where,
)
from zero_optax._types import ArrayLike, Tensor


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
            otherwise) of shape `[...]` or `[..., 1]`.
        targets: Binary target masks of shape matching `predictions`.
        smooth: Additive smoothing constant in numerator and denominator.
        apply_sigmoid: Whether to apply logistic sigmoid to `predictions` first.

    Returns:
        Binary Dice loss Tensor of shape `[...]` (batch dimensions only).
    """
    preds = asarray(predictions)
    targs = asarray(targets)
    default_axis: int | tuple[int, ...] | None = 0 if preds.ndim == 1 else None
    if preds.ndim == targs.ndim and preds.shape[-1] != 1:
        preds = expand_dims(preds, axis=-1)
        targs = expand_dims(targs, axis=-1)
    return dice_loss(
        preds,
        targs,
        smooth=smooth,
        apply_softmax=apply_sigmoid,
        reduction="mean",
        axis=default_axis,
    )


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
        apply_softmax: Whether to apply softmax (or sigmoid for 1-channel) along axis `-1`.
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
    if preds.ndim == targs.ndim - 1:
        preds = expand_dims(preds, axis=-1)
    if targs.ndim == preds.ndim - 1:
        targs = expand_dims(targs, axis=-1)

    if apply_softmax:
        probs = sigmoid(preds) if preds.shape[-1] == 1 else softmax(preds, axis=-1)
    else:
        probs = preds

    spatial_axes: int | tuple[int, ...] = (
        tuple(range(1, probs.ndim - 1)) if axis is None else axis
    )
    if spatial_axes != ():
        intersection = reduce_sum(probs * targs, axis=spatial_axes)
        pred_sum = reduce_sum(probs, axis=spatial_axes)
        target_sum = reduce_sum(targs, axis=spatial_axes)
    else:
        intersection = probs * targs
        pred_sum = probs
        target_sum = targs

    smooth_t = asarray(smooth)
    alpha_t = asarray(alpha)
    beta_t = asarray(beta)

    numer = intersection + smooth_t
    denom = (
        intersection
        + alpha_t * (pred_sum - intersection)
        + beta_t * (target_sum - intersection)
        + smooth_t
    )
    per_class_loss = 1.0 - (numer / denom)

    if class_weights is not None:
        per_class_loss = per_class_loss * asarray(class_weights)

    if ignore_background and probs.shape[-1] > 1:
        per_class_loss = per_class_loss[..., 1:]

    if reduction == "none":
        return per_class_loss
    if reduction == "sum":
        return reduce_sum(per_class_loss, axis=-1)
    return reduce_mean(per_class_loss, axis=-1)


def multiclass_generalized_dice_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    smooth: float | ArrayLike = 1.0,
    apply_softmax: bool = True,
    ignore_background: bool = False,
) -> Tensor:
    """Compute Multiclass Generalized Dice Loss weighted by inverse squared class frequency.

    Args:
        predictions: Model predictions of shape `[..., num_classes]`.
        targets: Target probabilities of shape `[..., num_classes]`.
        smooth: Smoothing constant added to numerator and denominator.
        apply_softmax: Whether to apply softmax to `predictions` along axis `-1`.
        ignore_background: If True, drop channel `0` before computing the loss.

    Returns:
        Scalar Generalized Dice loss Tensor averaged across all classes and batch.
    """
    preds = asarray(predictions)
    targs = asarray(targets)
    freq_axes = tuple(range(targs.ndim - 1))
    class_frequencies = reduce_sum(targs, axis=freq_axes) if freq_axes != () else targs
    raw_weights = 1.0 / (class_frequencies * class_frequencies + 1e-7)
    weight_sum = reduce_sum(raw_weights, axis=-1, keepdims=True)
    num_classes = float(targs.shape[-1])
    class_weights = (
        raw_weights / where(weight_sum == 0.0, 1.0, weight_sum) * num_classes
    )
    return reduce_mean(
        dice_loss(
            preds,
            targs,
            class_weights=class_weights,
            smooth=smooth,
            apply_softmax=apply_softmax,
            reduction="none",
            ignore_background=ignore_background,
        )
    )


__all__ = [
    "binary_dice_loss",
    "dice_loss",
    "multiclass_generalized_dice_loss",
]
