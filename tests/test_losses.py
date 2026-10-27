"""Comprehensive unit and gradient tests for `zero_optax.losses`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ml_switcheroo_compiler.core.tensor import Tensor

from tests.utils import assert_allclose_optax, numeric_grad
from zero_optax._tree import NEG_INF, asarray, reduce_max, reduce_sum, reshape
from zero_optax.losses import (
    binary_dice_loss,
    convex_kl_divergence,
    cosine_distance,
    cosine_similarity,
    ctc_loss,
    ctc_loss_with_forward_probs,
    dice_loss,
    hinge_loss,
    huber_loss,
    kl_divergence,
    kl_divergence_with_log_targets,
    l2_loss,
    log_cosh,
    make_fenchel_young_loss,
    multiclass_generalized_dice_loss,
    multiclass_hinge_loss,
    multiclass_perceptron_loss,
    multiclass_sparsemax_loss,
    ntxent,
    perceptron_loss,
    poly_loss_cross_entropy,
    ranking_softmax_loss,
    safe_softmax_cross_entropy,
    sigmoid_binary_cross_entropy,
    sigmoid_focal_loss,
    smooth_labels,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
    sparsemax_loss,
    squared_error,
    triplet_margin_loss,
)
from zero_optax.losses.sequence import _logaddexp


def test_regression_losses() -> None:
    """Test `squared_error`, `l2_loss`, `huber_loss`, `log_cosh`, and cosine losses with numerical parity and gradients."""
    pred = asarray([1.0, 2.0])
    targ = asarray([1.0, 1.0])
    j_pred = jnp.array([1.0, 2.0], dtype=jnp.float32)
    j_targ = jnp.array([1.0, 1.0], dtype=jnp.float32)

    assert_allclose_optax(squared_error(pred, targ), np.array([0.0, 1.0]))
    assert_allclose_optax(squared_error(pred), np.array([1.0, 4.0]))

    assert_allclose_optax(l2_loss(pred, targ), np.array([0.0, 0.5]))
    assert_allclose_optax(l2_loss(pred), np.array([0.5, 2.0]))

    assert_allclose_optax(huber_loss(pred, targ), np.array([0.0, 0.5]))
    assert_allclose_optax(huber_loss(pred), np.array([0.5, 1.5]))
    assert_allclose_optax(huber_loss(pred, targ, delta=0.5), np.array([0.0, 0.375]))

    assert_allclose_optax(log_cosh(pred, pred), np.array([0.0, 0.0]), atol=1e-6)
    assert_allclose_optax(log_cosh(pred), np.asarray(optax.log_cosh(j_pred)))
    assert_allclose_optax(
        log_cosh(pred, targ), np.asarray(optax.log_cosh(j_pred, j_targ))
    )

    v1 = asarray([[1.0, 0.0], [1.0, 1.0]])
    v2 = asarray([[1.0, 0.0], [-1.0, -1.0]])
    j_v1 = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    j_v2 = jnp.array([[1.0, 0.0], [-1.0, -1.0]], dtype=jnp.float32)

    sim = cosine_similarity(v1, v2)
    dist = cosine_distance(v1, v2)
    assert_allclose_optax(sim, np.asarray(optax.cosine_similarity(j_v1, j_v2)))
    assert_allclose_optax(dist, np.asarray(optax.cosine_distance(j_v1, j_v2)))

    mask = asarray([[True, False], [True, True]])
    j_mask = jnp.array([[True, False], [True, True]])
    sim_masked = cosine_similarity(v1, v2, where=mask)
    assert_allclose_optax(
        sim_masked, np.asarray(optax.cosine_similarity(j_v1, j_v2, where=j_mask))
    )

    def se_fn(x: Tensor) -> Tensor:
        """Sum of squared errors for gradient checking.

        Args:
            x: Input predictions Tensor.

        Returns:
            Scalar sum of squared errors.
        """
        return reduce_sum(squared_error(x, targ))

    def l2_fn(x: Tensor) -> Tensor:
        """Sum of L2 losses for gradient checking.

        Args:
            x: Input predictions Tensor.

        Returns:
            Scalar sum of L2 losses.
        """
        return reduce_sum(l2_loss(x, targ))

    def huber_fn(x: Tensor) -> Tensor:
        """Sum of Huber losses for gradient checking.

        Args:
            x: Input predictions Tensor.

        Returns:
            Scalar sum of Huber losses.
        """
        return reduce_sum(huber_loss(x, targ, delta=0.5))

    def log_cosh_fn(x: Tensor) -> Tensor:
        """Sum of log-cosh losses for gradient checking.

        Args:
            x: Input predictions Tensor.

        Returns:
            Scalar sum of log-cosh losses.
        """
        return reduce_sum(log_cosh(x, targ))

    def cos_dist_fn(x: Tensor) -> Tensor:
        """Sum of cosine distances for gradient checking.

        Args:
            x: Input predictions Tensor.

        Returns:
            Scalar sum of cosine distances.
        """
        return reduce_sum(cosine_distance(x, v2))

    assert_allclose_optax(numeric_grad(se_fn, pred), np.array([0.0, 2.0]))
    assert_allclose_optax(numeric_grad(l2_fn, pred), np.array([0.0, 1.0]))
    assert_allclose_optax(numeric_grad(huber_fn, pred), np.array([0.0, 0.5]))
    assert_allclose_optax(
        numeric_grad(log_cosh_fn, pred), np.tanh(np.array([0.0, 1.0])), rtol=1e-4
    )
    assert_allclose_optax(
        numeric_grad(cos_dist_fn, asarray([[1.0, 2.0], [2.0, 1.0]])),
        np.array(
            [
                [-0.8 / np.sqrt(5.0), 0.4 / np.sqrt(5.0)],
                [-0.2 / np.sqrt(10.0), 0.4 / np.sqrt(10.0)],
            ]
        ),
        rtol=1e-3,
        atol=1e-3,
    )


def test_classification_losses() -> None:
    """Test binary and multiclass classification losses and divergences against `optax`."""
    pred = asarray([0.5, -2.0])
    targ = asarray([1.0, -1.0])
    targ_01 = asarray([1.0, 0.0])
    j_pred = jnp.array([0.5, -2.0], dtype=jnp.float32)
    j_targ_01 = jnp.array([1.0, 0.0], dtype=jnp.float32)

    assert_allclose_optax(hinge_loss(pred, targ), np.array([0.5, 0.0]))
    assert_allclose_optax(perceptron_loss(pred, targ), np.array([0.0, 0.0]))

    assert_allclose_optax(
        sigmoid_binary_cross_entropy(pred, targ_01),
        np.asarray(optax.sigmoid_binary_cross_entropy(j_pred, j_targ_01)),
    )
    assert_allclose_optax(
        sigmoid_focal_loss(pred, targ_01),
        np.asarray(optax.losses.sigmoid_focal_loss(j_pred, j_targ_01)),
    )
    assert_allclose_optax(
        sigmoid_focal_loss(pred, targ_01, alpha=0.5),
        np.asarray(optax.losses.sigmoid_focal_loss(j_pred, j_targ_01, alpha=0.5)),
    )

    logits = asarray([[1.0, 2.0], [0.5, 0.5]])
    labels_float = asarray([[0.0, 1.0], [1.0, 0.0]])
    labels_int = asarray([1, 0])
    where_mask = asarray([[True, True], [False, False]])

    j_logits = jnp.array([[1.0, 2.0], [0.5, 0.5]], dtype=jnp.float32)
    j_labels_float = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    j_labels_int = jnp.array([1, 0], dtype=jnp.int32)
    j_where = jnp.array([[True, True], [False, False]])

    assert_allclose_optax(
        safe_softmax_cross_entropy(logits, labels_float),
        np.asarray(optax.safe_softmax_cross_entropy(j_logits, j_labels_float)),
    )
    assert_allclose_optax(
        softmax_cross_entropy(logits, labels_float),
        np.asarray(optax.softmax_cross_entropy(j_logits, j_labels_float)),
    )
    assert_allclose_optax(
        softmax_cross_entropy(logits, labels_float, where=where_mask),
        np.asarray(
            optax.softmax_cross_entropy(j_logits, j_labels_float, where=j_where)
        ),
    )
    assert_allclose_optax(
        softmax_cross_entropy_with_integer_labels(logits, labels_int),
        np.asarray(
            optax.softmax_cross_entropy_with_integer_labels(j_logits, j_labels_int)
        ),
    )
    assert_allclose_optax(
        softmax_cross_entropy_with_integer_labels(
            logits, labels_int, where=asarray([[True, True], [True, False]])
        ),
        np.asarray(
            optax.softmax_cross_entropy_with_integer_labels(
                j_logits, j_labels_int, where=jnp.array([[True, True], [True, False]])
            )
        ),
    )

    assert_allclose_optax(
        multiclass_hinge_loss(logits, labels_int),
        np.asarray(optax.losses.multiclass_hinge_loss(j_logits, j_labels_int)),
    )
    assert_allclose_optax(
        multiclass_perceptron_loss(logits, labels_int),
        np.asarray(optax.losses.multiclass_perceptron_loss(j_logits, j_labels_int)),
    )

    assert_allclose_optax(
        poly_loss_cross_entropy(logits, labels_float),
        np.asarray(optax.losses.poly_loss_cross_entropy(j_logits, j_labels_float)),
    )
    assert_allclose_optax(
        poly_loss_cross_entropy(logits, labels_float, where=where_mask),
        np.asarray(
            optax.losses.poly_loss_cross_entropy(
                j_logits, j_labels_float, where=j_where
            )
        ),
    )

    log_preds = asarray([[-0.69314718, -0.69314718], [-0.69314718, -0.69314718]])
    j_log_preds = jnp.array(
        [[-0.69314718, -0.69314718], [-0.69314718, -0.69314718]], dtype=jnp.float32
    )
    assert_allclose_optax(
        kl_divergence(log_preds, labels_float),
        np.asarray(optax.kl_divergence(j_log_preds, j_labels_float)),
    )
    assert_allclose_optax(
        kl_divergence(log_preds, labels_float, where=where_mask),
        np.asarray(optax.kl_divergence(j_log_preds, j_labels_float, where=j_where)),
    )
    assert_allclose_optax(
        kl_divergence_with_log_targets(log_preds, log_preds),
        np.asarray(
            optax.losses.kl_divergence_with_log_targets(j_log_preds, j_log_preds)
        ),
    )
    assert_allclose_optax(
        kl_divergence_with_log_targets(log_preds, log_preds, where=where_mask),
        np.asarray(
            optax.losses.kl_divergence_with_log_targets(
                j_log_preds, j_log_preds, where=j_where
            )
        ),
    )
    assert_allclose_optax(
        convex_kl_divergence(log_preds, labels_float),
        np.asarray(optax.convex_kl_divergence(j_log_preds, j_labels_float)),
    )
    assert_allclose_optax(
        convex_kl_divergence(log_preds, labels_float, where=where_mask),
        np.asarray(
            optax.convex_kl_divergence(j_log_preds, j_labels_float, where=j_where)
        ),
    )

    smoothed = smooth_labels(labels_float, alpha=0.1)
    assert_allclose_optax(
        smoothed, np.asarray(optax.smooth_labels(j_labels_float, alpha=0.1))
    )
    smoothed_all = smooth_labels(labels_float, alpha=0.1, axis=None)
    assert smoothed_all.shape == (2, 2)
    smoothed_tup = smooth_labels(labels_float, alpha=0.1, axis=(0, 1))
    assert smoothed_tup.shape == (2, 2)
    smoothed_where = smooth_labels(labels_float, alpha=0.1, where=where_mask)
    assert smoothed_where.shape == (2, 2)


def test_classification_edge_cases() -> None:
    """Test empty tensor shapes and axis reductions in classification losses."""
    empty_1d = asarray([])
    assert safe_softmax_cross_entropy(empty_1d, empty_1d).shape == ()
    assert softmax_cross_entropy(empty_1d, empty_1d, axis=None).shape == ()
    assert softmax_cross_entropy(empty_1d, empty_1d, where=empty_1d).shape == ()
    assert softmax_cross_entropy_with_integer_labels(empty_1d, empty_1d).shape == (0,)
    assert poly_loss_cross_entropy(empty_1d, empty_1d).shape == ()


def test_classification_gradients() -> None:
    """Test finite-difference gradients of classification and divergence losses."""
    pred = asarray([-0.5, 2.0])
    targ = asarray([1.0, -1.0])

    def h_fn(x: Tensor) -> Tensor:
        """Sum of hinge losses.

        Args:
            x: Predictions Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(hinge_loss(x, targ))

    assert_allclose_optax(numeric_grad(h_fn, pred), np.array([-1.0, 1.0]))

    def p_fn(x: Tensor) -> Tensor:
        """Sum of perceptron losses.

        Args:
            x: Predictions Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(perceptron_loss(x, targ))

    assert_allclose_optax(numeric_grad(p_fn, pred), np.array([-1.0, 1.0]))

    targ_bce = asarray([1.0, 0.0])

    def bce_fn(x: Tensor) -> Tensor:
        """Sum of sigmoid BCE losses.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(sigmoid_binary_cross_entropy(x, targ_bce))

    expected_grad = 1.0 / (1.0 + np.exp(-np.array([-0.5, 2.0]))) - np.array([1.0, 0.0])
    assert_allclose_optax(numeric_grad(bce_fn, pred), expected_grad, rtol=1e-4)

    def focal_fn(x: Tensor) -> Tensor:
        """Sum of sigmoid focal losses.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(sigmoid_focal_loss(x, targ_bce, gamma=0.0))

    assert_allclose_optax(numeric_grad(focal_fn, pred), expected_grad, rtol=1e-4)

    logits = asarray([[2.0, 1.0, 0.1], [0.5, -0.5, 2.0]])
    labels = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels_int = asarray([0, 1])

    def sf_fn(x: Tensor) -> Tensor:
        """Sum of safe softmax cross-entropy.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(safe_softmax_cross_entropy(x, labels))

    def ceil_fn(x: Tensor) -> Tensor:
        """Sum of integer-label softmax cross-entropy.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(softmax_cross_entropy_with_integer_labels(x, labels_int))

    def mc_hinge_fn(x: Tensor) -> Tensor:
        """Sum of multiclass hinge losses.

        Args:
            x: Scores Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(multiclass_hinge_loss(x, labels_int))

    def mc_perc_fn(x: Tensor) -> Tensor:
        """Sum of multiclass perceptron losses.

        Args:
            x: Scores Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(multiclass_perceptron_loss(x, labels_int))

    def poly_fn(x: Tensor) -> Tensor:
        """Sum of PolyLoss cross-entropy with epsilon=0.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(poly_loss_cross_entropy(x, labels, epsilon=0.0))

    def kl_fn(x: Tensor) -> Tensor:
        """Sum of KL divergence losses.

        Args:
            x: Log-predictions Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(kl_divergence(x, labels))

    def ckl_fn(x: Tensor) -> Tensor:
        """Sum of convex KL divergence losses.

        Args:
            x: Log-predictions Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(convex_kl_divergence(x, labels))

    assert_allclose_optax(numeric_grad(sf_fn, logits), numeric_grad(ceil_fn, logits))
    assert_allclose_optax(numeric_grad(poly_fn, logits), numeric_grad(sf_fn, logits))
    assert_allclose_optax(
        numeric_grad(mc_hinge_fn, asarray([[2.5, 1.0, 0.1], [0.5, -0.5, 2.0]])),
        np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0]]),
    )
    assert_allclose_optax(
        numeric_grad(mc_perc_fn, logits),
        np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0]]),
    )
    assert_allclose_optax(
        numeric_grad(kl_fn, logits), -np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    assert_allclose_optax(
        numeric_grad(ckl_fn, logits),
        np.exp(np.array([[2.0, 1.0, 0.1], [0.5, -0.5, 2.0]]))
        - np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        rtol=1e-4,
    )


def test_dice_losses() -> None:
    """Test `binary_dice_loss`, `dice_loss`, and `multiclass_generalized_dice_loss` values and gradients."""
    preds = asarray([[2.0, -2.0], [-2.0, 2.0]])
    targs = asarray([[1.0, 0.0], [0.0, 1.0]])

    s_pos = 1.0 / (1.0 + np.exp(-2.0))
    s_neg = 1.0 / (1.0 + np.exp(2.0))
    expected_b_dice = 1.0 - (s_pos + 1.0) / (
        s_pos + 0.5 * s_neg + 0.5 * (1.0 - s_pos) + 1.0
    )
    b_dice = binary_dice_loss(preds, targs)
    assert_allclose_optax(b_dice, np.array([expected_b_dice, expected_b_dice]))

    b_dice_1ch = binary_dice_loss(asarray([[2.0], [-2.0]]), asarray([[1.0], [0.0]]))
    assert b_dice_1ch.shape == (2,)

    probs_in = asarray([[0.9, 0.1], [0.1, 0.9]])
    b_dice_prob = binary_dice_loss(probs_in, targs, apply_sigmoid=False)
    expected_prob_dice = 1.0 - (0.9 + 1.0) / (0.9 + 0.5 * 0.1 + 0.5 * 0.1 + 1.0)
    assert_allclose_optax(
        b_dice_prob, np.array([expected_prob_dice, expected_prob_dice])
    )

    d_expand_p = dice_loss(asarray([2.0, -2.0]), asarray([[1.0], [0.0]]))
    d_expand_t = dice_loss(asarray([[2.0], [-2.0]]), asarray([1.0, 0.0]))
    assert_allclose_optax(d_expand_p, d_expand_t)

    d_mean = dice_loss(probs_in, targs, apply_softmax=False, reduction="mean")
    c0_loss = 1.0 - (0.9 + 1.0) / (0.9 + 0.5 * 0.0 + 0.5 * 0.1 + 1.0)
    c1_loss = 1.0 - (0.0 + 1.0) / (0.0 + 0.5 * 0.1 + 0.5 * 0.0 + 1.0)
    expected_d_mean = 0.5 * (c0_loss + c1_loss)
    assert_allclose_optax(d_mean, np.array([expected_d_mean, expected_d_mean]))

    d_sum = dice_loss(probs_in, targs, reduction="sum", apply_softmax=False)
    assert_allclose_optax(d_sum, np.array([c0_loss + c1_loss, c0_loss + c1_loss]))

    d_none = dice_loss(
        probs_in,
        targs,
        apply_softmax=False,
        reduction="none",
        ignore_background=True,
        class_weights=asarray([0.5, 1.5]),
        axis=0,
    )
    expected_c1_axis0 = 1.5 * (1.0 - (0.9 + 1.0) / (0.9 + 0.5 * 0.1 + 0.5 * 0.1 + 1.0))
    assert_allclose_optax(d_none, np.array([expected_c1_axis0]))

    d_weighted_mean = dice_loss(
        probs_in,
        targs,
        apply_softmax=False,
        reduction="mean",
        class_weights=asarray([1.0, 2.0]),
    )
    assert_allclose_optax(
        d_weighted_mean,
        np.array(
            [
                0.5 * (1.0 * c0_loss + 2.0 * c1_loss),
                0.5 * (1.0 * c1_loss + 2.0 * c0_loss),
            ]
        ),
    )

    preds_3d = asarray([[[[1.0, 2.0], [2.0, 1.0]]]])
    targs_3d = asarray([[[[0.0, 1.0], [1.0, 0.0]]]])
    d_3d = dice_loss(preds_3d, targs_3d)
    p_hi = np.exp(2.0) / (np.exp(1.0) + np.exp(2.0))
    p_lo = 1.0 - p_hi
    exp_3d = 1.0 - (p_hi + 1.0) / (p_hi + 0.5 * p_lo + 0.5 * p_lo + 1.0)
    assert_allclose_optax(d_3d, np.array([exp_3d]))

    gdl_3d = multiclass_generalized_dice_loss(
        asarray([[[[0.2, 0.8], [0.8, 0.2]]]]),
        targs_3d,
        apply_softmax=False,
        ignore_background=True,
    )
    exp_gdl_3d = 1.0 - (0.8 + 1.0) / (0.8 + 0.5 * 0.2 + 0.5 * 0.2 + 1.0)
    assert_allclose_optax(gdl_3d, np.array(exp_gdl_3d))

    gdl_1d = multiclass_generalized_dice_loss(
        asarray([0.8, 0.2]), asarray([1.0, 0.0]), apply_softmax=False
    )
    assert gdl_1d.shape == ()

    b_dice_1d = binary_dice_loss(asarray([2.0, -2.0]), asarray([1.0, 0.0]))
    assert_allclose_optax(b_dice_1d, np.array(expected_b_dice))

    def b_dice_sum(x: Tensor) -> Tensor:
        """Sum of binary dice loss for gradient check.

        Args:
            x: Predictions Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(binary_dice_loss(x, targs, apply_sigmoid=False))

    grad_bd = numeric_grad(b_dice_sum, probs_in)
    denom_val = 0.9 + 0.5 * 0.1 + 0.5 * 0.1 + 1.0
    numer_val = 0.9 + 1.0
    d_pos = -(denom_val - 0.5 * numer_val) / (denom_val**2)
    d_neg = (0.5 * numer_val) / (denom_val**2)
    assert_allclose_optax(
        grad_bd,
        np.array([[d_pos, d_neg], [d_neg, d_pos]]),
        rtol=1e-4,
        atol=1e-4,
    )

    with pytest.raises(ValueError, match="Unsupported reduction"):
        dice_loss(preds, targs, reduction="invalid")


def test_metric_losses() -> None:
    """Test `triplet_margin_loss` and `ntxent` values and gradients against `optax`."""
    anchors = asarray([[0.0, 0.0], [1.0, 1.0]])
    positives = asarray([[0.1, 0.1], [1.2, 1.0]])
    negatives = asarray([[2.0, 2.0], [1.05, 1.0]])

    t_loss = triplet_margin_loss(anchors, positives, negatives)
    pos_d1 = np.sqrt(0.2**2 + 1e-6)
    neg_d1 = np.sqrt(0.05**2 + 1e-6)
    assert_allclose_optax(t_loss, np.array([0.0, pos_d1 - neg_d1 + 1.0]), rtol=1e-5)

    def trip_fn(x: Tensor) -> Tensor:
        """Sum of triplet margin loss for gradient checking.

        Args:
            x: Anchors Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(triplet_margin_loss(x, positives, negatives))

    expected_trip_grad = np.array(
        [[0.0, 0.0], [(-0.2 / pos_d1) - (-0.05 / neg_d1), 0.0]]
    )
    assert_allclose_optax(
        numeric_grad(trip_fn, anchors),
        expected_trip_grad,
        rtol=1e-3,
        atol=1e-3,
    )

    emb = asarray([[1.0, 0.2], [0.9, 0.1], [-0.1, 1.0], [0.1, 0.9]])
    lbl = asarray([0, 0, 1, 1])
    j_emb = jnp.array(
        [[1.0, 0.2], [0.9, 0.1], [-0.1, 1.0], [0.1, 0.9]], dtype=jnp.float32
    )
    j_lbl = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    c_loss = ntxent(emb, lbl, temperature=0.1)
    assert_allclose_optax(
        c_loss,
        np.asarray(optax.losses.ntxent(j_emb, j_lbl, temperature=0.1)),
        rtol=1e-4,
        atol=1e-4,
    )

    def ntxent_fn(x: Tensor) -> Tensor:
        """Evaluate ntxent loss for gradient checking.

        Args:
            x: Embeddings Tensor.

        Returns:
            Scalar ntxent loss.
        """
        return ntxent(x, lbl, temperature=0.5)

    def j_ntxent_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate optax ntxent loss for JAX reference gradient.

        Args:
            x: JAX embeddings array.

        Returns:
            Scalar JAX loss.
        """
        return jnp.asarray(optax.losses.ntxent(x, j_lbl, temperature=0.5))

    ngrad_ntxent = numeric_grad(ntxent_fn, emb)
    jgrad_ntxent = np.asarray(jax.grad(j_ntxent_fn)(j_emb))
    assert_allclose_optax(ngrad_ntxent, jgrad_ntxent, rtol=1e-3, atol=1e-3)

    lbl_unique = asarray([0, 1, 2, 3])
    assert float(ntxent(emb, lbl_unique).item()) == 0.0

    lbl_1d_emb = asarray([1.0, 2.0])
    assert_allclose_optax(
        ntxent(lbl_1d_emb, asarray([0, 0]), temperature=1.0),
        np.array(0.0),
        atol=1e-5,
    )

    with pytest.raises(ValueError, match="same leading dimension"):
        ntxent(emb, asarray([0, 1]))


def test_sequence_ctc_losses() -> None:
    """Test `ctc_loss` and `ctc_loss_with_forward_probs` values and gradients against `optax`."""
    raw_logits = [
        [[0.1, 2.0, 0.5], [2.0, 0.1, 0.2], [0.1, 0.2, 2.5]],
        [[1.0, 0.5, 0.2], [0.2, 1.5, 0.3], [0.3, 1.5, 0.1]],
    ]
    raw_lpad = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    raw_labels = [[1, 2], [1, 1]]
    raw_labpad = [[0.0, 0.0], [0.0, 1.0]]

    logits = asarray(raw_logits)
    logit_paddings = asarray(raw_lpad)
    labels = asarray(raw_labels)
    label_paddings = asarray(raw_labpad)

    loss_val, logalpha_blank, logalpha_nonblank = ctc_loss_with_forward_probs(
        logits, logit_paddings, labels, label_paddings
    )
    o_loss, o_blank, o_nonblank = optax.ctc_loss_with_forward_probs(
        jnp.array(raw_logits, dtype=jnp.float32),
        jnp.array(raw_lpad, dtype=jnp.float32),
        jnp.array(raw_labels, dtype=jnp.int32),
        jnp.array(raw_labpad, dtype=jnp.float32),
    )
    assert_allclose_optax(loss_val, np.asarray(o_loss), rtol=1e-4, atol=1e-4)
    assert_allclose_optax(logalpha_blank, np.asarray(o_blank), rtol=1e-4, atol=1e-4)
    assert_allclose_optax(
        logalpha_nonblank, np.asarray(o_nonblank), rtol=1e-4, atol=1e-4
    )

    direct_loss = ctc_loss(logits, logit_paddings, labels, label_paddings)
    assert_allclose_optax(loss_val, direct_loss)

    single_label_loss = ctc_loss(
        logits,
        logit_paddings,
        asarray([[1], [2]]),
        asarray([[0.0], [0.0]]),
    )
    o_single_loss = optax.ctc_loss(
        jnp.array(raw_logits, dtype=jnp.float32),
        jnp.array(raw_lpad, dtype=jnp.float32),
        jnp.array([[1], [2]], dtype=jnp.int32),
        jnp.array([[0.0], [0.0]], dtype=jnp.float32),
    )
    assert_allclose_optax(single_label_loss, np.asarray(o_single_loss), rtol=1e-4)

    def ctc_sum_fn(x: Tensor) -> Tensor:
        """Sum of CTC losses for finite-difference gradient check.

        Args:
            x: Logits Tensor.

        Returns:
            Scalar sum of CTC losses.
        """
        return reduce_sum(ctc_loss(x, logit_paddings, labels, label_paddings))

    def j_ctc_sum_fn(x: jnp.ndarray) -> jnp.ndarray:
        """Sum of optax CTC losses for JAX reference gradient.

        Args:
            x: JAX logits array.

        Returns:
            Scalar sum of JAX CTC losses.
        """
        return jnp.sum(
            optax.ctc_loss(
                x,
                jnp.array(raw_lpad, dtype=jnp.float32),
                jnp.array(raw_labels, dtype=jnp.int32),
                jnp.array(raw_labpad, dtype=jnp.float32),
            )
        )

    assert_allclose_optax(
        numeric_grad(ctc_sum_fn, logits),
        np.asarray(jax.grad(j_ctc_sum_fn)(jnp.array(raw_logits, dtype=jnp.float32))),
        rtol=1e-3,
        atol=1e-3,
    )

    empty_time_logits = reshape(asarray([]), (1, 0, 2))
    empty_time_pad = reshape(asarray([]), (1, 0))
    loss_empty, _, _ = ctc_loss_with_forward_probs(
        empty_time_logits, empty_time_pad, asarray([[1]]), asarray([[0.0]])
    )
    assert_allclose_optax(loss_empty, np.array([100000.0]))
    assert float(_logaddexp(asarray(NEG_INF), asarray(NEG_INF)).item()) == NEG_INF


def test_sparsemax_ranking_and_fenchel_young() -> None:
    """Test `sparsemax_loss`, `multiclass_sparsemax_loss`, `ranking_softmax_loss`, and `make_fenchel_young_loss` against `optax`."""
    raw_logits = [[-2.0, 0.0, 2.0], [0.5, -0.5, 0.0]]
    raw_bin = [[0, 1, 1], [1, 0, 0]]
    logits = asarray(raw_logits)
    labels_bin = asarray(raw_bin)
    j_logits = jnp.array(raw_logits, dtype=jnp.float32)
    j_bin = jnp.array(raw_bin, dtype=jnp.int32)

    sp_loss = sparsemax_loss(logits, labels_bin)
    assert_allclose_optax(
        sp_loss, np.asarray(optax.losses.sparsemax_loss(j_logits, j_bin))
    )

    mc_sp = multiclass_sparsemax_loss(logits, asarray([2, 0]))
    assert_allclose_optax(
        mc_sp,
        np.asarray(
            optax.losses.multiclass_sparsemax_loss(
                j_logits, jnp.array([2, 0], dtype=jnp.int32)
            )
        ),
    )

    def mc_sp_fn(x: Tensor) -> Tensor:
        """Sum of multiclass sparsemax loss for gradient check.

        Args:
            x: Scores Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(multiclass_sparsemax_loss(x, asarray([2, 0])))

    ngrad_mc_sp = numeric_grad(mc_sp_fn, logits)
    assert_allclose_optax(ngrad_mc_sp[0], np.array([0.0, 0.0, 0.0]), atol=1e-4)
    assert_allclose_optax(
        ngrad_mc_sp[1], np.array([-0.25, 0.0, 0.25]), rtol=1e-3, atol=1e-3
    )

    rank_logits = asarray([[1.0, 2.0], [0.5, 0.5]])
    rank_labels = asarray([[0.0, 1.0], [1.0, 0.0]])
    j_rank_logits = jnp.array([[1.0, 2.0], [0.5, 0.5]], dtype=jnp.float32)
    j_rank_labels = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    assert_allclose_optax(
        ranking_softmax_loss(rank_logits, rank_labels),
        np.asarray(optax.losses.ranking_softmax_loss(j_rank_logits, j_rank_labels)),
    )
    assert_allclose_optax(
        ranking_softmax_loss(rank_logits, rank_labels, reduce_fn=None),
        np.asarray(
            optax.losses.ranking_softmax_loss(
                j_rank_logits, j_rank_labels, reduce_fn=None
            )
        ),
    )
    assert_allclose_optax(
        ranking_softmax_loss(
            rank_logits,
            rank_labels,
            where=asarray([[True, True], [False, False]]),
            weights=asarray([[1.0, 2.0], [1.0, 1.0]]),
        ),
        np.asarray(
            optax.losses.ranking_softmax_loss(
                j_rank_logits,
                j_rank_labels,
                where=jnp.array([[True, True], [False, False]]),
                weights=jnp.array([[1.0, 2.0], [1.0, 1.0]], dtype=jnp.float32),
            )
        ),
    )
    assert_allclose_optax(
        ranking_softmax_loss(
            rank_logits,
            rank_labels,
            where=asarray([[True, True], [False, False]]),
            reduce_fn=reduce_sum,
        ),
        np.asarray(
            optax.losses.ranking_softmax_loss(
                j_rank_logits,
                j_rank_labels,
                where=jnp.array([[True, True], [False, False]]),
                reduce_fn=jnp.sum,
            )
        ),
    )

    def max_fn_axis(
        s: Tensor,
        /,
        *args: float | int | str | bool,
        **kwargs: float | int | str | bool,
    ) -> Tensor:
        """Maximum along the last axis.

        Args:
            s: Scores Tensor.
            *args: Extra positional arguments.
            **kwargs: Extra keyword arguments.

        Returns:
            Reduced maximum Tensor.
        """
        del args, kwargs
        return reduce_max(s, axis=-1)

    fy_scores = asarray([[1.0, 3.0], [0.5, 2.0]])
    fy_targets = asarray([[0.8, 0.2], [0.6, 0.4]])

    fy_axis = make_fenchel_young_loss(max_fn_axis)
    res_axis = fy_axis(fy_scores, fy_targets)
    assert_allclose_optax(res_axis, np.array([1.6, 0.9]))

    def max_fn_1d(
        s: Tensor,
        /,
        *args: float | int | str | bool,
        **kwargs: float | int | str | bool,
    ) -> Tensor:
        """Global maximum (returns scalar even for 2D input, triggering row fallback).

        Args:
            s: Scores Tensor.
            *args: Extra positional arguments.
            **kwargs: Extra keyword arguments.

        Returns:
            Scalar maximum Tensor.
        """
        del args, kwargs
        return reduce_max(s)

    fy_1d = make_fenchel_young_loss(max_fn_1d)
    res_1d = fy_1d(fy_scores, fy_targets)
    assert_allclose_optax(res_1d, np.array([1.6, 0.9]))

    def fy_sum_fn(x: Tensor) -> Tensor:
        """Sum of Fenchel-Young losses for gradient check.

        Args:
            x: Scores Tensor.

        Returns:
            Scalar sum.
        """
        return reduce_sum(fy_1d(x, fy_targets))

    assert_allclose_optax(
        numeric_grad(fy_sum_fn, fy_scores),
        np.array([[-0.8, 0.8], [-0.6, 0.6]]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_losses_chex_validation_and_zero_jax_inputs() -> None:
    """Verify `zero_chex` shape/rank assertions in losses and direct `zero_jax` array inputs."""
    from zero_flax.types import ensure_array

    pred = ensure_array([[1.0, 2.0], [3.0, 4.0]])
    targ = ensure_array([[1.5, 2.5], [2.5, 3.5]])
    se = squared_error(pred, targ)
    assert se.shape == (2, 2)

    with pytest.raises(AssertionError):
        squared_error(asarray([1.0, 2.0]), asarray([1.0, 2.0, 3.0]))
    with pytest.raises(AssertionError):
        l2_loss(asarray([1.0, 2.0]), asarray([1.0, 2.0, 3.0]))
    with pytest.raises(AssertionError):
        kl_divergence(asarray([0.1, 0.2]), asarray([0.1, 0.2, 0.7]))
    with pytest.raises(AssertionError):
        convex_kl_divergence(asarray([0.1, 0.2]), asarray([0.1, 0.2, 0.7]))
    with pytest.raises(AssertionError):
        multiclass_hinge_loss(asarray([[1.0, 2.0]]), asarray([[0]]))
    with pytest.raises(AssertionError):
        ranking_softmax_loss(asarray([1.0, 2.0]), asarray([1.0, 2.0, 3.0]))


def test_losses_zero_jax_grad_and_jit_differentiability() -> None:
    """Verify all 17 loss functions and elementary `_tree.py` math differentiate under `zero_jax.grad` and `@zero_jax.jit`."""
    import math
    from collections.abc import Callable

    import zero_jax
    import zero_jax.numpy as jnp_zero
    from zero_jax._types import PyTree as JaxPyTree
    from zero_jax._types import VariadicCallable
    from zero_jax.numpy import ndarray as JaxArray

    from zero_optax._tree import (
        abs_val,
        clip,
        concatenate,
        cos,
        exp,
        floor,
        log,
        log1p,
        log_softmax,
        maximum,
        mean_axis,
        minimum,
        power,
        reduce_max,
        reduce_mean,
        reduce_min,
        reduce_sum,
        reshape,
        sign,
        sin,
        sqrt,
        squeeze,
        stack,
        sum_to_Float,
        where,
        zeros,
    )
    from zero_optax._types import ArrayLike
    from zero_optax.losses import smooth_l1_loss

    def _jnp(val: ArrayLike) -> JaxArray:
        """Convert `val` into a narrowed `zero_jax.numpy.ndarray`.

        Args:
            val: Input array-like value.

        Returns:
            Converted `JaxArray`.
        """
        arr = jnp_zero.asarray(asarray(val))
        assert isinstance(arr, JaxArray)
        return arr

    # Test smooth_l1_loss with targets=None and explicit targets
    sl1_none = smooth_l1_loss(asarray([-2.0, 0.5, 2.0]), alpha=1.0)
    assert_allclose_optax(sl1_none, np.array([1.5, 0.125, 1.5]))
    sl1_targ = smooth_l1_loss(asarray([1.0, 2.0]), asarray([0.5, 0.0]), alpha=1.0)
    assert_allclose_optax(sl1_targ, np.array([0.125, 1.5]))

    # Test ctc_loss_with_forward_probs when max_label_len == 0 (_concat_head_and_tail total_cols <= 1)
    ctc_empty_lbl_loss, _, _ = ctc_loss_with_forward_probs(
        asarray([[[0.2, 0.8]]]),
        asarray([[0.0]]),
        zeros((1, 0), dtype=int),
        zeros((1, 0)),
    )
    assert ctc_empty_lbl_loss.shape == (1,)

    # Test elementary _tree.py math on JaxArray directly and under zero_jax.grad
    jx = _jnp([0.5, 1.5])
    assert abs_val(jx).shape == (2,)
    assert exp(jx).shape == (2,)
    assert log(jx).shape == (2,)
    assert log1p(jx).shape == (2,)
    assert sqrt(jx).shape == (2,)
    assert sign(jx).shape == (2,)
    assert cos(jx).shape == (2,)
    assert sin(jx).shape == (2,)
    assert floor(jx).shape == (2,)
    assert clip(jx, a_min=0.6, a_max=1.2).shape == (2,)
    assert clip(jx).shape == (2,)
    assert sum_to_Float(jx).shape == ()
    assert mean_axis(jx).shape == ()
    assert asarray(jx, dtype="int32").shape == (2,)
    assert asarray([jx, jx]).shape == (2, 2)
    assert sin(asarray([0.5, 1.0])).shape == (2,)
    assert concatenate([asarray([1.0]), asarray([2.0])], axis=0).shape == (2,)

    class _PureToList:
        """Minimal `ToListArrayProtocol` implementation without `.tolist()`."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return shape `(2,)`.

            Returns:
                Shape tuple `(2,)`.
            """
            return (2,)

        @property
        def dtype(self) -> str:
            """Return `'float32'`.

            Returns:
                String `'float32'`.
            """
            return "float32"

        def to_list(self) -> list[float | complex]:
            """Return flat list with a complex and float element.

            Returns:
                List `[1.0, 2.0 + 0.0j]`.
            """
            return [1.0, complex(2.0, 0.0)]

    assert asarray(_PureToList()).shape == (2,)

    def _composite_math_loss(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Exercise VJP rules for mean, min, minimum, maximum, power, squeeze, and non-diff ops.

        Args:
            *args: Positional arguments `(v,)`.
            **kwargs: Unused keyword arguments.

        Returns:
            Scalar objective `Tensor`.
        """
        del kwargs
        v = args[0]
        assert isinstance(v, (JaxArray, Tensor))
        m_all = reduce_mean(v)
        m_tup = reduce_sum(v, axis=(0, 1), keepdims=False)
        m_ax = reduce_mean(v, axis=-1, keepdims=False)
        min_ax = reduce_min(v, axis=-1, keepdims=False)
        max_ax = reduce_max(v, axis=-1, keepdims=False)
        clamped = clip(v, 0.1, 2.5)
        pow_const = power(clamped, 2.0)
        pow_var = power(clamped, clamped)
        sq = squeeze(reshape(v, (2, 1, 3)), axis=1)
        casted = asarray(v, dtype="float32")
        st_neg = reduce_sum(stack([v, v], axis=-1), axis=-1)
        cond_Part = where(
            asarray(v) > 0.0,
            pow_const + pow_var + sq + casted + st_neg,
            minimum(v, maximum(v, 0.0)),
        )
        non_diff = sign(v) + floor(v)
        return (
            m_all
            + m_tup
            + reduce_sum(m_ax + min_ax + max_ax)
            + reduce_sum(cond_Part + non_diff)
        )

    logits = _jnp([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    labels = _jnp([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    int_labels = _jnp([0, 1])
    bin_targets = _jnp([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    log_p = _jnp(log_softmax(logits, axis=-1))
    emb = _jnp([[1.0, 0.2, 0.3], [0.5, 0.8, 0.2], [0.2, 0.3, 0.9], [0.1, 0.9, 0.4]])
    emb_labels = _jnp([0, 0, 1, 1])

    ctc_logits = _jnp([[[0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.7, 0.2, 0.1]]])
    ctc_logit_pad = _jnp([[0.0, 0.0, 0.0]])
    ctc_labels = _jnp([[1, 2]])
    ctc_label_pad = _jnp([[0.0, 0.0]])

    grad_comp = zero_jax.grad(_composite_math_loss)(logits)
    assert isinstance(grad_comp, JaxArray)
    assert grad_comp.shape == (2, 3)

    loss_cases: list[tuple[Callable[[JaxArray], Tensor], JaxArray]] = [
        (lambda x: reduce_sum(softmax_cross_entropy(x, labels)), logits),
        (
            lambda x: reduce_sum(
                softmax_cross_entropy_with_integer_labels(x, int_labels)
            ),
            logits,
        ),
        (lambda x: reduce_sum(sigmoid_binary_cross_entropy(x, labels)), logits),
        (lambda x: reduce_sum(hinge_loss(x, bin_targets)), logits),
        (lambda x: reduce_sum(multiclass_hinge_loss(x, int_labels)), logits),
        (lambda x: reduce_sum(kl_divergence(x, labels)), log_p),
        (lambda x: reduce_sum(convex_kl_divergence(x, labels)), log_p),
        (
            lambda x: reduce_sum(ctc_loss(x, ctc_logit_pad, ctc_labels, ctc_label_pad)),
            ctc_logits,
        ),
        (lambda x: reduce_sum(l2_loss(x, labels)), logits),
        (lambda x: reduce_sum(squared_error(x, labels)), logits),
        (lambda x: reduce_sum(huber_loss(x, labels)), logits),
        (lambda x: reduce_sum(smooth_l1_loss(x, labels)), logits),
        (lambda x: reduce_sum(log_cosh(x, labels)), logits),
        (lambda x: reduce_sum(cosine_similarity(x, labels)), logits),
        (lambda x: reduce_sum(cosine_distance(x, labels)), logits),
        (lambda x: reduce_sum(ranking_softmax_loss(x, labels)), logits),
        (lambda x: reduce_sum(ntxent(x, emb_labels, temperature=0.5)), emb),
    ]

    def _make_wrapped_loss(
        fn: Callable[[JaxArray], Tensor],
    ) -> VariadicCallable[JaxPyTree]:
        """Build a `VariadicCallable[JaxPyTree]` wrapper around `fn`.

        Args:
            fn: Loss callable taking a single `JaxArray`.

        Returns:
            Wrapped callable compatible with `zero_jax.value_and_grad` and `zero_jax.jit`.
        """

        def _wrapped_loss(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
            """Evaluate `fn` on `args[0]`.

            Args:
                *args: Single positional input `JaxArray`.
                **kwargs: Unused keyword arguments.

            Returns:
                Scalar loss `Tensor`.
            """
            del kwargs
            inp = args[0]
            assert isinstance(inp, JaxArray)
            res: Tensor = fn(inp)
            return res

        return _wrapped_loss

    for loss_fn, x_in in loss_cases:
        wrapped_fn = _make_wrapped_loss(loss_fn)
        val, g = zero_jax.value_and_grad(wrapped_fn)(x_in)
        assert isinstance(val, (Tensor, JaxArray))
        assert isinstance(g, JaxArray)
        assert g.shape == x_in.shape
        g_t = asarray(g)
        assert float(reduce_sum(g_t * g_t).item()) > 0.0
        jit_val = zero_jax.jit(wrapped_fn)(x_in)
        assert isinstance(jit_val, (Tensor, JaxArray))
        assert math.isclose(
            float(asarray(val).item()),
            float(asarray(jit_val).item()),
            rel_tol=1e-5,
        )
