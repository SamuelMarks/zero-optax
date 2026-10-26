"""Reference equivalence tests comparing `zero_optax` outputs against upstream `optax`."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import optax

import zero_optax
from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax._tree import asarray, logsumexp
from zero_optax._types import Tensor
from zero_optax.base import GradientTransformation, OptState


def _zero_logsumexp_max_fun(
    scores: Tensor,
    /,
    *args: float | int | str | bool,
    **kwargs: float | int | str | bool,
) -> Tensor:
    """Compute zero_optax logsumexp along axis -1 for Fenchel-Young parity testing.

    Args:
        scores: Input Tensor.
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        Reduced Tensor.
    """
    del args, kwargs
    return logsumexp(scores, axis=-1)


def test_reference_losses_parity() -> None:
    """Compare all 26 loss functions between `zero_optax` and `optax`."""
    preds_1d = [1.0, 2.0, -0.5]
    targs_1d = [1.5, 1.0, 0.5]
    z_preds_1d = asarray(preds_1d)
    z_targs_1d = asarray(targs_1d)
    j_preds_1d = jnp.array(preds_1d, dtype=jnp.float32)
    j_targs_1d = jnp.array(targs_1d, dtype=jnp.float32)

    assert_allclose_optax(
        zero_optax.squared_error(z_preds_1d, z_targs_1d),
        np.asarray(optax.squared_error(j_preds_1d, j_targs_1d)),
    )
    assert_allclose_optax(
        zero_optax.l2_loss(z_preds_1d, z_targs_1d),
        np.asarray(optax.l2_loss(j_preds_1d, j_targs_1d)),
    )
    assert_allclose_optax(
        zero_optax.huber_loss(z_preds_1d, z_targs_1d, delta=0.8),
        np.asarray(optax.huber_loss(j_preds_1d, j_targs_1d, delta=0.8)),
    )
    assert_allclose_optax(
        zero_optax.log_cosh(z_preds_1d, z_targs_1d),
        np.asarray(optax.log_cosh(j_preds_1d, j_targs_1d)),
    )
    assert_allclose_optax(
        zero_optax.cosine_similarity(z_preds_1d, z_targs_1d),
        np.asarray(optax.cosine_similarity(j_preds_1d, j_targs_1d)),
    )
    assert_allclose_optax(
        zero_optax.cosine_distance(z_preds_1d, z_targs_1d),
        np.asarray(optax.cosine_distance(j_preds_1d, j_targs_1d)),
    )

    bin_targs = [1.0, -1.0, 1.0]
    assert_allclose_optax(
        zero_optax.hinge_loss(z_preds_1d, asarray(bin_targs)),
        np.asarray(optax.hinge_loss(j_preds_1d, jnp.array(bin_targs))),
    )
    assert_allclose_optax(
        zero_optax.perceptron_loss(z_preds_1d, asarray(bin_targs)),
        np.asarray(optax.losses.perceptron_loss(j_preds_1d, jnp.array(bin_targs))),
    )

    bce_targs = [1.0, 0.0, 1.0]
    assert_allclose_optax(
        zero_optax.sigmoid_binary_cross_entropy(z_preds_1d, asarray(bce_targs)),
        np.asarray(
            optax.sigmoid_binary_cross_entropy(j_preds_1d, jnp.array(bce_targs))
        ),
    )
    assert_allclose_optax(
        zero_optax.sigmoid_focal_loss(
            z_preds_1d, asarray(bce_targs), alpha=0.25, gamma=2.0
        ),
        np.asarray(
            optax.losses.sigmoid_focal_loss(
                j_preds_1d, jnp.array(bce_targs), alpha=0.25, gamma=2.0
            )
        ),
    )

    logits_2d = [[1.0, 2.0, 0.5], [0.2, -1.0, 1.5]]
    probs_2d = [[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]
    int_labels = [1, 2]
    z_logits = asarray(logits_2d)
    z_probs = asarray(probs_2d)
    z_int = asarray(int_labels)
    j_logits = jnp.array(logits_2d, dtype=jnp.float32)
    j_probs = jnp.array(probs_2d, dtype=jnp.float32)
    j_int = jnp.array(int_labels, dtype=jnp.int32)

    assert_allclose_optax(
        zero_optax.safe_softmax_cross_entropy(z_logits, z_probs),
        np.asarray(optax.safe_softmax_cross_entropy(j_logits, j_probs)),
    )
    assert_allclose_optax(
        zero_optax.softmax_cross_entropy(z_logits, z_probs),
        np.asarray(optax.softmax_cross_entropy(j_logits, j_probs)),
    )
    assert_allclose_optax(
        zero_optax.softmax_cross_entropy_with_integer_labels(z_logits, z_int),
        np.asarray(optax.softmax_cross_entropy_with_integer_labels(j_logits, j_int)),
    )
    assert_allclose_optax(
        zero_optax.poly_loss_cross_entropy(z_logits, z_probs, epsilon=1.5),
        np.asarray(
            optax.losses.poly_loss_cross_entropy(j_logits, j_probs, epsilon=1.5)
        ),
    )
    assert_allclose_optax(
        zero_optax.multiclass_hinge_loss(z_logits, z_int),
        np.asarray(optax.losses.multiclass_hinge_loss(j_logits, j_int)),
    )
    assert_allclose_optax(
        zero_optax.multiclass_perceptron_loss(z_logits, z_int),
        np.asarray(optax.losses.multiclass_perceptron_loss(j_logits, j_int)),
    )

    log_probs_2d = np.log(np.array(probs_2d, dtype=np.float32))
    z_log_probs = asarray(log_probs_2d.tolist())
    j_log_probs = jnp.array(log_probs_2d, dtype=jnp.float32)
    assert_allclose_optax(
        zero_optax.kl_divergence(z_log_probs, z_probs),
        np.asarray(optax.kl_divergence(j_log_probs, j_probs)),
        atol=1e-5,
    )
    assert_allclose_optax(
        zero_optax.kl_divergence_with_log_targets(z_log_probs, z_log_probs),
        np.asarray(
            optax.losses.kl_divergence_with_log_targets(j_log_probs, j_log_probs)
        ),
        atol=1e-5,
    )
    assert_allclose_optax(
        zero_optax.convex_kl_divergence(z_log_probs, z_probs),
        np.asarray(optax.convex_kl_divergence(j_log_probs, j_probs)),
        atol=1e-5,
    )
    assert_allclose_optax(
        zero_optax.smooth_labels(z_probs, alpha=0.2),
        np.asarray(optax.smooth_labels(j_probs, alpha=0.2)),
    )

    assert_allclose_optax(
        zero_optax.sparsemax_loss(z_logits, asarray([[0, 1, 0], [1, 0, 1]])),
        np.asarray(
            optax.losses.sparsemax_loss(j_logits, jnp.array([[0, 1, 0], [1, 0, 1]]))
        ),
    )
    assert_allclose_optax(
        zero_optax.multiclass_sparsemax_loss(z_logits, z_int),
        np.asarray(optax.losses.multiclass_sparsemax_loss(j_logits, j_int)),
    )
    assert_allclose_optax(
        zero_optax.ranking_softmax_loss(z_logits, z_probs),
        np.asarray(optax.losses.ranking_softmax_loss(j_logits, j_probs)),
    )

    z_fy = zero_optax.make_fenchel_young_loss(_zero_logsumexp_max_fun)
    o_fy = optax.losses.make_fenchel_young_loss(jax_logsumexp)
    assert_allclose_optax(
        z_fy(z_logits, z_probs),
        np.asarray(o_fy(j_logits, j_probs)),
    )

    dice_probs = asarray([[0.8, 0.2], [0.1, 0.9]])
    dice_targs = asarray([[1.0, 0.0], [0.0, 1.0]])
    expected_bd = 1.0 - (0.8 + 1.0) / (0.8 + 0.5 * 0.2 + 0.5 * 0.2 + 1.0)
    expected_bd2 = 1.0 - (0.9 + 1.0) / (0.9 + 0.5 * 0.1 + 0.5 * 0.1 + 1.0)
    assert_allclose_optax(
        zero_optax.binary_dice_loss(dice_probs, dice_targs, apply_sigmoid=False),
        np.array([expected_bd, expected_bd2]),
    )
    assert_allclose_optax(
        zero_optax.dice_loss(dice_probs, dice_targs, apply_softmax=False),
        np.array(
            [
                0.5 * ((1.0 - 1.8 / 1.9) + (1.0 - 1.0 / 1.1)),
                0.5 * ((1.0 - 1.0 / 1.05) + (1.0 - 1.9 / 1.95)),
            ]
        ),
    )
    gdl_val = zero_optax.multiclass_generalized_dice_loss(
        dice_probs, dice_targs, apply_softmax=False
    )
    assert float(gdl_val.item()) > 0.0

    anc = [[0.0, 0.0], [1.0, 1.0]]
    pos = [[0.1, 0.1], [1.2, 1.0]]
    neg = [[2.0, 2.0], [1.05, 1.0]]
    pos_d1 = np.sqrt(0.2**2 + 1e-6)
    neg_d1 = np.sqrt(0.05**2 + 1e-6)
    assert_allclose_optax(
        zero_optax.triplet_margin_loss(asarray(anc), asarray(pos), asarray(neg)),
        np.array([0.0, pos_d1 - neg_d1 + 1.0]),
        rtol=1e-5,
    )

    emb = [[1.0, 0.2], [0.9, 0.1], [-0.1, 1.0], [0.1, 0.9]]
    lbl = [0, 0, 1, 1]
    assert_allclose_optax(
        zero_optax.ntxent(asarray(emb), asarray(lbl), temperature=0.1),
        np.asarray(
            optax.losses.ntxent(
                jnp.array(emb, dtype=jnp.float32),
                jnp.array(lbl, dtype=jnp.int32),
                temperature=0.1,
            )
        ),
        rtol=1e-4,
        atol=1e-4,
    )

    ctc_logits = [[[0.1, 2.0, 0.5], [2.0, 0.1, 0.2], [0.1, 0.2, 2.5]]]
    ctc_lpad = [[0.0, 0.0, 0.0]]
    ctc_labels = [[1, 2]]
    ctc_labpad = [[0.0, 0.0]]
    z_ctc, z_b, z_nb = zero_optax.ctc_loss_with_forward_probs(
        asarray(ctc_logits),
        asarray(ctc_lpad),
        asarray(ctc_labels),
        asarray(ctc_labpad),
    )
    o_ctc, o_b, o_nb = optax.ctc_loss_with_forward_probs(
        jnp.array(ctc_logits),
        jnp.array(ctc_lpad),
        jnp.array(ctc_labels),
        jnp.array(ctc_labpad),
    )
    assert_allclose_optax(z_ctc, np.asarray(o_ctc), rtol=1e-4, atol=1e-4)
    assert_allclose_optax(z_b, np.asarray(o_b), rtol=1e-4, atol=1e-4)
    assert_allclose_optax(z_nb, np.asarray(o_nb), rtol=1e-4, atol=1e-4)
    assert_allclose_optax(
        zero_optax.ctc_loss(
            asarray(ctc_logits),
            asarray(ctc_lpad),
            asarray(ctc_labels),
            asarray(ctc_labpad),
        ),
        np.asarray(o_ctc),
        rtol=1e-4,
        atol=1e-4,
    )


def jax_logsumexp(
    scores: jnp.ndarray,
    /,
    *args: float | int | str | bool,
    **kwargs: float | int | str | bool,
) -> jnp.ndarray:
    """Compute JAX logsumexp along axis -1 for Fenchel-Young parity testing.

    Args:
        scores: Input JAX array.
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        Reduced JAX array.
    """
    del args, kwargs
    return jnp.log(jnp.sum(jnp.exp(scores), axis=-1))


def test_reference_schedules_parity() -> None:
    """Compare all 14 learning rate schedules between `zero_optax` and `optax`."""
    schedule_pairs = [
        (zero_optax.constant_schedule(0.05), optax.constant_schedule(0.05)),
        (
            zero_optax.linear_schedule(0.1, 0.001, 20, transition_begin=5),
            optax.linear_schedule(0.1, 0.001, 20, transition_begin=5),
        ),
        (
            zero_optax.polynomial_schedule(0.1, 0.001, 2.0, 20, transition_begin=5),
            optax.polynomial_schedule(0.1, 0.001, 2.0, 20, transition_begin=5),
        ),
        (
            zero_optax.exponential_decay(
                0.1, 10, 0.8, transition_begin=2, staircase=True, end_value=0.01
            ),
            optax.exponential_decay(
                0.1, 10, 0.8, transition_begin=2, staircase=True, end_value=0.01
            ),
        ),
        (
            zero_optax.cosine_decay_schedule(0.1, 30, alpha=0.1, exponent=1.5),
            optax.cosine_decay_schedule(0.1, 30, alpha=0.1, exponent=1.5),
        ),
        (
            zero_optax.join_schedules(
                [
                    zero_optax.linear_schedule(0.0, 0.1, 10),
                    zero_optax.constant_schedule(0.05),
                ],
                [10],
            ),
            optax.join_schedules(
                [optax.linear_schedule(0.0, 0.1, 10), optax.constant_schedule(0.05)],
                [10],
            ),
        ),
        (
            zero_optax.piecewise_constant_schedule(0.1, {10: 0.5, 25: 0.2}),
            optax.piecewise_constant_schedule(0.1, {10: 0.5, 25: 0.2}),
        ),
        (
            zero_optax.piecewise_interpolate_schedule(
                "linear", 0.1, {10: 0.5, 25: 0.2}
            ),
            optax.piecewise_interpolate_schedule("linear", 0.1, {10: 0.5, 25: 0.2}),
        ),
        (
            zero_optax.piecewise_interpolate_schedule(
                "cosine", 0.1, {10: 0.5, 25: 0.2}
            ),
            optax.piecewise_interpolate_schedule("cosine", 0.1, {10: 0.5, 25: 0.2}),
        ),
        (
            zero_optax.warmup_constant_schedule(0.0, 0.1, 10),
            optax.warmup_constant_schedule(0.0, 0.1, 10),
        ),
        (
            zero_optax.warmup_cosine_decay_schedule(0.0, 0.1, 10, 40, end_value=0.001),
            optax.warmup_cosine_decay_schedule(0.0, 0.1, 10, 40, end_value=0.001),
        ),
        (
            zero_optax.warmup_exponential_decay_schedule(0.0, 0.1, 10, 15, 0.8),
            optax.warmup_exponential_decay_schedule(0.0, 0.1, 10, 15, 0.8),
        ),
        (
            zero_optax.cosine_onecycle_schedule(50, 0.1),
            optax.cosine_onecycle_schedule(50, 0.1),
        ),
        (
            zero_optax.linear_onecycle_schedule(50, 0.1),
            optax.linear_onecycle_schedule(50, 0.1),
        ),
        (
            zero_optax.sgdr_schedule(
                [
                    {
                        "init_value": 0.0,
                        "peak_value": 0.1,
                        "warmup_steps": 5,
                        "decay_steps": 15,
                        "end_value": 0.0,
                    },
                    {
                        "init_value": 0.0,
                        "peak_value": 0.05,
                        "warmup_steps": 5,
                        "decay_steps": 25,
                        "end_value": 0.0,
                    },
                ]
            ),
            optax.sgdr_schedule(
                [
                    {
                        "init_value": 0.0,
                        "peak_value": 0.1,
                        "warmup_steps": 5,
                        "decay_steps": 15,
                        "end_value": 0.0,
                    },
                    {
                        "init_value": 0.0,
                        "peak_value": 0.05,
                        "warmup_steps": 5,
                        "decay_steps": 25,
                        "end_value": 0.0,
                    },
                ]
            ),
        ),
    ]

    for z_sched, o_sched in schedule_pairs:
        for step in (0, 1, 5, 10, 15, 25, 40, 60):
            z_raw = z_sched(step)
            z_val = float(z_raw.item()) if isinstance(z_raw, Tensor) else float(z_raw)
            o_val = float(np.asarray(o_sched(step)).item())
            assert math.isclose(z_val, o_val, rel_tol=1e-5, abs_tol=1e-6)


def test_reference_optimizers_parity() -> None:
    """Compare multi-step optimizer updates between `zero_optax` and `optax`."""
    z_params = {"w": asarray([1.0, -2.0]), "b": asarray([0.5])}
    z_grads = {"w": asarray([0.1, -0.2]), "b": asarray([0.05])}
    j_params = {
        "w": jnp.array([1.0, -2.0], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    j_grads = {
        "w": jnp.array([0.1, -0.2], dtype=jnp.float32),
        "b": jnp.array([0.05], dtype=jnp.float32),
    }

    pairs: list[
        tuple[GradientTransformation[OptState], optax.GradientTransformation]
    ] = [
        (zero_optax.sgd(0.1, momentum=0.9), optax.sgd(0.1, momentum=0.9)),
        (zero_optax.adagrad(0.1), optax.adagrad(0.1)),
        (zero_optax.adam(0.1), optax.adam(0.1)),
        (zero_optax.adamw(0.1, weight_decay=1e-3), optax.adamw(0.1, weight_decay=1e-3)),
        (zero_optax.lamb(0.1, weight_decay=1e-3), optax.lamb(0.1, weight_decay=1e-3)),
        (zero_optax.lars(0.1, weight_decay=1e-3), optax.lars(0.1, weight_decay=1e-3)),
        (zero_optax.lion(0.1, weight_decay=1e-3), optax.lion(0.1, weight_decay=1e-3)),
        (zero_optax.adabelief(0.1), optax.adabelief(0.1)),
        (zero_optax.fromage(0.1), optax.fromage(0.1)),
        (
            zero_optax.novograd(0.1, weight_decay=1e-3),
            optax.novograd(0.1, weight_decay=1e-3),
        ),
        (
            zero_optax.rmsprop(0.1, centered=True, momentum=0.9),
            optax.rmsprop(0.1, centered=True, momentum=0.9),
        ),
        (zero_optax.yogi(0.1), optax.yogi(0.1)),
        (zero_optax.adamax(0.1), optax.adamax(0.1)),
        (
            zero_optax.adamaxw(0.1, weight_decay=1e-3),
            optax.adamaxw(0.1, weight_decay=1e-3),
        ),
        (zero_optax.amsgrad(0.1), optax.amsgrad(0.1)),
        (zero_optax.radam(0.1), optax.radam(0.1)),
        (zero_optax.rprop(0.1), optax.rprop(0.1)),
        (zero_optax.sm3(0.1), optax.sm3(0.1)),
        (zero_optax.sign_sgd(0.1), optax.sign_sgd(0.1)),
        (
            zero_optax.optimistic_gradient_descent(0.1),
            optax.optimistic_gradient_descent(0.1),
        ),
    ]

    for z_opt, o_opt in pairs:
        z_state = z_opt.init(z_params)
        o_state = o_opt.init(j_params)
        for _ in range(2):
            z_u, z_state = z_opt.update(z_grads, z_state, z_params)
            o_u, o_state = o_opt.update(j_grads, o_state, j_params)
            assert_allclose_optax(
                get_tree_item(z_u, "w"),
                np.asarray(o_u["w"]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert_allclose_optax(
                get_tree_item(z_u, "b"),
                np.asarray(o_u["b"]),
                rtol=1e-4,
                atol=1e-4,
            )

    # Rprop sign-flip step parity test
    z_rp = zero_optax.rprop(0.1)
    o_rp = optax.rprop(0.1)
    z_rp_st = z_rp.init(z_params)
    o_rp_st = o_rp.init(j_params)
    z_rp_u0, z_rp_st = z_rp.update(z_grads, z_rp_st, z_params)
    o_rp_u0, o_rp_st = o_rp.update(j_grads, o_rp_st, j_params)
    assert_allclose_optax(
        get_tree_item(z_rp_u0, "w"), np.asarray(o_rp_u0["w"]), rtol=1e-4, atol=1e-4
    )
    z_flip_grads = {"w": asarray([-0.1, 0.2]), "b": asarray([-0.05])}
    j_flip_grads = {
        "w": jnp.array([-0.1, 0.2], dtype=jnp.float32),
        "b": jnp.array([-0.05], dtype=jnp.float32),
    }
    z_rp_u1, _ = z_rp.update(z_flip_grads, z_rp_st, z_params)
    o_rp_u1, _ = o_rp.update(j_flip_grads, o_rp_st, j_params)
    assert_allclose_optax(
        get_tree_item(z_rp_u1, "w"), np.asarray(o_rp_u1["w"]), rtol=1e-4, atol=1e-4
    )

    # Adafactor factored 2D parity test (min_dim_size_to_factor=2)
    z_p2d = {"m": asarray([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]])}
    z_g2d = {"m": asarray([[0.1, -0.2, 0.05], [-0.15, 0.25, 0.1]])}
    j_p2d = {"m": jnp.array([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]], dtype=jnp.float32)}
    j_g2d = {"m": jnp.array([[0.1, -0.2, 0.05], [-0.15, 0.25, 0.1]], dtype=jnp.float32)}

    for z_af, o_af in (
        (
            zero_optax.adafactor(learning_rate=0.1, min_dim_size_to_factor=2),
            optax.adafactor(learning_rate=0.1, min_dim_size_to_factor=2),
        ),
        (
            zero_optax.sm3(0.1, momentum=0.9),
            optax.sm3(0.1, momentum=0.9),
        ),
    ):
        z_st = z_af.init(z_p2d)
        o_st = o_af.init(j_p2d)
        for _ in range(2):
            z_u2, z_st = z_af.update(z_g2d, z_st, z_p2d)
            o_u2, o_st = o_af.update(j_g2d, o_st, j_p2d)
            assert_allclose_optax(
                get_tree_item(z_u2, "m"),
                np.asarray(o_u2["m"]),
                rtol=1e-4,
                atol=1e-4,
            )

    # Masked global norm parity test: masked leaves must not contribute to global norm
    mask_dict = {"w": True, "b": False}
    z_mclip = zero_optax.masked(zero_optax.clip_by_global_norm(0.05), mask_dict)
    o_mclip = optax.masked(optax.clip_by_global_norm(0.05), mask_dict)
    z_mc_u, _ = z_mclip.update(
        {"w": asarray([3.0, 4.0]), "b": asarray([100.0])},
        z_mclip.init(z_params),
        z_params,
    )
    o_mc_u, _ = o_mclip.update(
        {
            "w": jnp.array([3.0, 4.0], dtype=jnp.float32),
            "b": jnp.array([100.0], dtype=jnp.float32),
        },
        o_mclip.init(j_params),
        j_params,
    )
    assert_allclose_optax(
        get_tree_item(z_mc_u, "w"), np.asarray(o_mc_u["w"]), rtol=1e-5, atol=1e-5
    )
    assert_allclose_optax(
        get_tree_item(z_mc_u, "b"), np.asarray(o_mc_u["b"]), rtol=1e-5, atol=1e-5
    )
