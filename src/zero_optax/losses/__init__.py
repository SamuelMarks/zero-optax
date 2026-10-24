"""zero_optax losses package."""

from __future__ import annotations

from zero_optax.losses.classification import (
    convex_kl_divergence,
    hinge_loss,
    kl_divergence,
    kl_divergence_with_log_targets,
    multiclass_hinge_loss,
    multiclass_perceptron_loss,
    perceptron_loss,
    poly_loss_cross_entropy,
    safe_softmax_cross_entropy,
    sigmoid_binary_cross_entropy,
    sigmoid_focal_loss,
    smooth_labels,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
)
from zero_optax.losses.dice import (
    binary_dice_loss,
    dice_loss,
    multiclass_generalized_dice_loss,
)
from zero_optax.losses.fenchel_young import MaxFun, make_fenchel_young_loss
from zero_optax.losses.metric import ntxent, triplet_margin_loss
from zero_optax.losses.ranking import ranking_softmax_loss
from zero_optax.losses.regression import (
    cosine_distance,
    cosine_similarity,
    huber_loss,
    l2_loss,
    log_cosh,
    squared_error,
)
from zero_optax.losses.sequence import ctc_loss, ctc_loss_with_forward_probs
from zero_optax.losses.sparsemax import multiclass_sparsemax_loss, sparsemax_loss

__all__ = [
    "MaxFun",
    "binary_dice_loss",
    "convex_kl_divergence",
    "cosine_distance",
    "cosine_similarity",
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "dice_loss",
    "hinge_loss",
    "huber_loss",
    "kl_divergence",
    "kl_divergence_with_log_targets",
    "l2_loss",
    "log_cosh",
    "make_fenchel_young_loss",
    "multiclass_generalized_dice_loss",
    "multiclass_hinge_loss",
    "multiclass_perceptron_loss",
    "multiclass_sparsemax_loss",
    "ntxent",
    "perceptron_loss",
    "poly_loss_cross_entropy",
    "ranking_softmax_loss",
    "safe_softmax_cross_entropy",
    "sigmoid_binary_cross_entropy",
    "sigmoid_focal_loss",
    "smooth_labels",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "sparsemax_loss",
    "squared_error",
    "triplet_margin_loss",
]
