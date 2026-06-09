"""zero_optax losses package."""

from zero_optax.losses.regression import squared_error, l2_loss, huber_loss
from zero_optax.losses.classification import (
    hinge_loss,
    perceptron_loss,
    safe_softmax_cross_entropy,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
    sigmoid_binary_cross_entropy,
    sigmoid_focal_loss,
    multiclass_hinge_loss,
    multiclass_perceptron_loss,
    poly_loss_cross_entropy,
)
from zero_optax.losses.sequence import ctc_loss, ctc_loss_with_forward_probs
from zero_optax.losses.ranking import ranking_softmax_loss
from zero_optax.losses.sparsemax import sparsemax_loss, multiclass_sparsemax_loss
from zero_optax.losses.fenchel_young import make_fenchel_young_loss

__all__ = [
    "squared_error",
    "l2_loss",
    "huber_loss",
    "hinge_loss",
    "perceptron_loss",
    "safe_softmax_cross_entropy",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "sigmoid_binary_cross_entropy",
    "sigmoid_focal_loss",
    "multiclass_hinge_loss",
    "multiclass_perceptron_loss",
    "poly_loss_cross_entropy",
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "ranking_softmax_loss",
    "sparsemax_loss",
    "multiclass_sparsemax_loss",
    "make_fenchel_young_loss",
]
