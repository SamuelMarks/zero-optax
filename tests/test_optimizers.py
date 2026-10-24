"""Comprehensive unit tests for `zero_optax.optimizers`, `transform`, `update`, and `clipping`."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax._tree import (
    _resolve_dtype,
    apply_mask_tree,
    arange,
    asarray,
    leaf_l2_norm,
    logsumexp,
    ones,
    ones_like,
    squeeze_axes,
    tree_l2_norm,
    tree_map,
    tree_zeros_like,
)
from zero_optax._types import DType, Params, PyTree, TensorLeaf
from zero_optax.base import EmptyState, ScaleState, identity, set_to_zero
from zero_optax.clipping import adaptive_grad_clip, clip, clip_by_global_norm
from zero_optax.combine import chain
from zero_optax.optimizers import (
    adabelief,
    adagrad,
    adam,
    adamw,
    fromage,
    lamb,
    lars,
    lion,
    novograd,
    rmsprop,
    sgd,
    yogi,
)
from zero_optax.optimizers.adabelief import _as_belief_state
from zero_optax.optimizers.adagrad import _as_rss_state
from zero_optax.optimizers.adam import _as_adam_state
from zero_optax.optimizers.lars import _as_trace_state
from zero_optax.optimizers.lion import _as_lion_state
from zero_optax.optimizers.novograd import _as_novograd_state
from zero_optax.optimizers.rmsprop import _as_rms_inner_state, _as_trace_or_empty
from zero_optax.schedules import constant_schedule
from zero_optax.transform import (
    add_decayed_weights,
    scale,
    scale_by_adam,
    scale_by_belief,
    scale_by_learning_rate,
    scale_by_lion,
    scale_by_novograd,
    scale_by_rms,
    scale_by_rss,
    scale_by_schedule,
    scale_by_stddev,
    scale_by_trust_ratio,
    scale_by_yogi,
    trace,
)
from zero_optax.update import (
    apply_updates,
    global_norm,
    incremental_update,
    periodic_update,
)


class _SamplePair(NamedTuple):
    """NamedTuple container for testing NamedTuple preservation in `tree_map`.

    Attributes:
        left: First leaf element.
        right: Second leaf element.
    """

    left: TensorLeaf
    right: TensorLeaf


def _make_params() -> dict[str, PyTree]:
    """Create a sample parameter dictionary PyTree.

    Returns:
        Dictionary with `'w'` and `'b'` Tensor leaves.
    """
    return {"w": asarray([1.0, 2.0]), "b": asarray([0.5])}


def _make_updates() -> dict[str, PyTree]:
    """Create a sample gradient update dictionary PyTree.

    Returns:
        Dictionary with `'w'` and `'b'` Tensor leaves.
    """
    return {"w": asarray([0.1, -0.1]), "b": asarray([0.05])}


def _true_mask(p: Params) -> dict[str, bool]:
    """Return a mask dictionary mapping every key in `p` to `True`.

    Args:
        p: Parameters PyTree.

    Returns:
        Mask dictionary `{'w': True, 'b': True}`.
    """
    del p
    return {"w": True, "b": True}


def test_tree_and_helper_branches() -> None:
    """Test all container, dtype, and edge-case branches in `_tree.py` and state narrowers."""
    assert _resolve_dtype(None) is None
    assert _resolve_dtype(DType.float32) == DType.float32
    assert _resolve_dtype("float32") == DType.float32
    assert _resolve_dtype(float) == DType.float32
    assert _resolve_dtype(int) == DType.int32
    assert _resolve_dtype(bool) == DType("bool")

    t_ones = ones(2)
    assert_allclose_optax(t_ones, np.array([1.0, 1.0]))
    assert_allclose_optax(ones_like(t_ones), np.array([1.0, 1.0]))
    assert_allclose_optax(arange(3), np.array([0, 1, 2]))
    assert_allclose_optax(arange(1, 4, 1), np.array([1, 2, 3]))

    def _double(x: TensorLeaf) -> TensorLeaf:
        """Double a leaf value if non-None.

        Args:
            x: Input leaf.

        Returns:
            Doubled leaf.
        """
        return None if x is None else asarray(x) * 2.0

    tup_tree: PyTree = (asarray([1.0]), None)
    list_tree: PyTree = [asarray([2.0]), None]
    mapped_tup = tree_map(_double, tup_tree)
    mapped_list = tree_map(_double, list_tree)
    co_tup = tree_map(lambda a, b: a, (1.0, 2.0), (3.0, 4.0))
    assert isinstance(mapped_tup, tuple)
    assert isinstance(mapped_list, list)
    assert isinstance(co_tup, tuple)
    zero_tup = tree_zeros_like(tup_tree)
    assert isinstance(zero_tup, tuple) and zero_tup[1] is None
    assert float(leaf_l2_norm(None).item()) == 0.0
    assert float(tree_l2_norm(tup_tree, squared=True).item()) == 1.0

    with pytest.raises(ValueError, match="Mapping"):
        tree_map(lambda a, b: a, {"w": 1.0}, {"other": 1.0})
    with pytest.raises(ValueError, match="sequence"):
        tree_map(lambda a, b: a, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="leaf"):
        tree_map(lambda a, b: a, 1.0, [1.0])

    assert apply_mask_tree((1.0, 2.0), (3.0, 4.0), (True, False)) == (1.0, 4.0)
    assert apply_mask_tree([1.0, 2.0], [3.0, 4.0], [False, True]) == [3.0, 2.0]
    assert apply_mask_tree(1.0, 2.0, {"unused": True}) == 1.0

    lse = logsumexp(asarray([[0.0, 0.0]]), axis=-1, keepdims=True)
    assert_allclose_optax(lse, np.array([[np.log(2.0)]]))
    lse_sq = logsumexp(asarray([[0.0, 0.0]]), axis=-1, keepdims=False)
    assert_allclose_optax(lse_sq, np.array([np.log(2.0)]))
    assert_allclose_optax(squeeze_axes(asarray([[5.0]]), None), np.array(5.0))

    for narrow_fn in (
        _as_belief_state,
        _as_rss_state,
        _as_adam_state,
        _as_trace_state,
        _as_lion_state,
        _as_novograd_state,
        _as_rms_inner_state,
        _as_trace_or_empty,
    ):
        with pytest.raises(TypeError):
            narrow_fn(ScaleState())


def test_base_combine_update_and_clipping() -> None:
    """Test `identity`, `set_to_zero`, `chain`, `update.py`, and `clipping.py`."""
    params = _make_params()
    updates = _make_updates()

    id_tx = identity()
    u_id, _ = id_tx.update(updates, id_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_id, "w"), np.array([0.1, -0.1]))

    zero_tx = set_to_zero()
    u_zero, _ = zero_tx.update(updates, zero_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_zero, "w"), np.array([0.0, 0.0]))

    chained = chain(sgd(0.1), sgd(0.2))
    c_state = chained.init(params)
    u_chain, _ = chained.update(updates, c_state, params)
    assert_allclose_optax(
        get_tree_item(u_chain, "w"), np.array([0.002, -0.002]), rtol=1e-5
    )
    with pytest.raises(ValueError, match="Chain state length mismatch"):
        chained.update(updates, EmptyState(), params)

    updated_p = apply_updates(
        {"w": asarray([1.0, 2.0]), "n": None},
        {"w": asarray([0.5, -0.5]), "n": None},
    )
    assert_allclose_optax(get_tree_item(updated_p, "w"), np.array([1.5, 1.5]))

    inc_p = incremental_update(
        {"w": asarray([2.0]), "n": None},
        {"w": asarray([1.0]), "n": None},
        0.5,
    )
    assert_allclose_optax(get_tree_item(inc_p, "w"), np.array([1.5]))

    per_p0 = periodic_update(
        {"w": asarray([2.0]), "n": None},
        {"w": asarray([1.0]), "n": None},
        asarray(4),
        2,
    )
    per_p1 = periodic_update(
        {"w": asarray([2.0]), "n": None},
        {"w": asarray([1.0]), "n": None},
        3,
        2,
    )
    assert_allclose_optax(get_tree_item(per_p0, "w"), np.array([2.0]))
    assert_allclose_optax(get_tree_item(per_p1, "w"), np.array([1.0]))

    assert_allclose_optax(global_norm(updates), np.sqrt(0.1**2 + (-0.1) ** 2 + 0.05**2))

    clip_tx = clip(0.02)
    u_clip, _ = clip_tx.update(
        {"w": asarray([0.1, -0.1]), "n": None},
        clip_tx.init(params),
    )
    assert_allclose_optax(get_tree_item(u_clip, "w"), np.array([0.02, -0.02]))

    gn_tx = clip_by_global_norm(0.01)
    u_gn, _ = gn_tx.update(
        {"w": asarray([1.0, 0.0]), "n": None},
        gn_tx.init(params),
    )
    assert_allclose_optax(get_tree_item(u_gn, "w"), np.array([0.01, 0.0]))

    agc_tx = adaptive_grad_clip(0.01)
    u_agc, _ = agc_tx.update(
        {"w": asarray([10.0, 0.0]), "n": None},
        agc_tx.init(params),
        {"w": asarray([1.0, 0.0]), "n": None},
    )
    assert_allclose_optax(get_tree_item(u_agc, "w"), np.array([0.01, 0.0]))
    with pytest.raises(ValueError, match="adaptive_grad_clip"):
        agc_tx.update(updates, agc_tx.init(params), None)


def test_transform_primitives() -> None:
    """Test all `scale_by_*`, `trace`, and `add_decayed_weights` primitives against `optax`."""
    params: dict[str, PyTree] = {"w": asarray([1.0, 2.0]), "n": None}
    updates: dict[str, PyTree] = {"w": asarray([0.2, -0.4]), "n": None}
    j_params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    j_updates = {"w": jnp.array([0.2, -0.4], dtype=jnp.float32)}

    tx_pairs = [
        (scale(-0.1), optax.scale(-0.1)),
        (
            scale_by_schedule(constant_schedule(-0.1)),
            optax.scale_by_schedule(optax.constant_schedule(-0.1)),
        ),
        (scale_by_learning_rate(0.1), optax.scale_by_learning_rate(0.1)),
        (
            scale_by_learning_rate(constant_schedule(0.1), flip_sign=False),
            optax.scale_by_learning_rate(optax.constant_schedule(0.1), flip_sign=False),
        ),
        (
            trace(0.9, nesterov=False, accumulator_dtype="float32"),
            optax.trace(0.9, nesterov=False, accumulator_dtype=jnp.float32),
        ),
        (trace(0.9, nesterov=True), optax.trace(0.9, nesterov=True)),
        (scale_by_rss(), optax.scale_by_rss()),
        (
            scale_by_adam(mu_dtype="float32", nesterov=False),
            optax.scale_by_adam(mu_dtype=jnp.float32, nesterov=False),
        ),
        (scale_by_adam(nesterov=True), optax.scale_by_adam(nesterov=True)),
        (scale_by_belief(nesterov=False), optax.scale_by_belief(nesterov=False)),
        (scale_by_belief(nesterov=True), optax.scale_by_belief(nesterov=True)),
        (
            scale_by_lion(mu_dtype="float32"),
            optax.scale_by_lion(mu_dtype=jnp.float32),
        ),
        (
            scale_by_rms(bias_correction=False, eps_in_sqrt=True),
            optax.scale_by_rms(bias_correction=False, eps_in_sqrt=True),
        ),
        (
            scale_by_rms(bias_correction=True, eps_in_sqrt=False),
            optax.scale_by_rms(bias_correction=True, eps_in_sqrt=False),
        ),
        (
            scale_by_stddev(bias_correction=False, eps_in_sqrt=True),
            optax.scale_by_stddev(bias_correction=False, eps_in_sqrt=True),
        ),
        (
            scale_by_stddev(initial_scale=0.1, bias_correction=True, eps_in_sqrt=False),
            optax.scale_by_stddev(
                initial_scale=0.1, bias_correction=True, eps_in_sqrt=False
            ),
        ),
        (scale_by_yogi(), optax.scale_by_yogi()),
        (
            scale_by_novograd(weight_decay=0.0, mu_dtype="float32"),
            optax.scale_by_novograd(weight_decay=0.0, mu_dtype=jnp.float32),
        ),
        (scale_by_trust_ratio(), optax.scale_by_trust_ratio()),
        (
            add_decayed_weights(0.01, mask={"w": True, "n": False}),
            optax.add_decayed_weights(0.01, mask={"w": True}),
        ),
    ]

    for z_tx, o_tx in tx_pairs:
        st = z_tx.init(params)
        ost = o_tx.init(j_params)
        u1, st1 = z_tx.update(updates, st, params)
        ou1, ost1 = o_tx.update(j_updates, ost, j_params)
        u2, _ = z_tx.update(updates, st1, params)
        ou2, _ = o_tx.update(j_updates, ost1, j_params)
        assert_allclose_optax(
            get_tree_item(u1, "w"), np.asarray(ou1["w"]), rtol=1e-4, atol=1e-4
        )
        assert_allclose_optax(
            get_tree_item(u2, "w"), np.asarray(ou2["w"]), rtol=1e-4, atol=1e-4
        )
        u_fallback, _ = z_tx.update(updates, EmptyState(), params)
        assert_allclose_optax(
            get_tree_item(u_fallback, "w"),
            np.asarray(ou1["w"]),
            rtol=1e-4,
            atol=1e-4,
        )

    with pytest.raises(ValueError, match="scale_by_novograd"):
        scale_by_novograd(weight_decay=0.1).update(updates, EmptyState(), None)
    u_ng_none, _ = scale_by_novograd(weight_decay=0.0).update(
        updates, EmptyState(), None
    )
    ou_ng_none, _ = optax.scale_by_novograd(weight_decay=0.0).update(
        j_updates, optax.scale_by_novograd(weight_decay=0.0).init(j_params), j_params
    )
    assert_allclose_optax(
        get_tree_item(u_ng_none, "w"), np.asarray(ou_ng_none["w"]), rtol=1e-4
    )
    u_aw_nomask, _ = add_decayed_weights(0.01, mask=None).update(
        updates, EmptyState(), params
    )
    assert_allclose_optax(
        get_tree_item(u_aw_nomask, "w"), np.array([0.21, -0.38]), rtol=1e-5
    )
    with pytest.raises(ValueError, match="scale_by_trust_ratio"):
        scale_by_trust_ratio().update(updates, EmptyState(), None)
    with pytest.raises(ValueError, match="add_decayed_weights"):
        add_decayed_weights(0.1).update(updates, EmptyState(), None)


def test_all_12_optimizers() -> None:
    """Test all 12 optimizers across momentum, Nesterov, masks, schedules, and error branches against `optax`."""
    params: dict[str, PyTree] = {
        "w": asarray([1.0, 2.0]),
        "b": asarray([0.5]),
        "n": None,
    }
    updates: dict[str, PyTree] = {
        "w": asarray([0.1, -0.1]),
        "b": asarray([0.05]),
        "n": None,
    }
    j_params = {
        "w": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    j_updates = {
        "w": jnp.array([0.1, -0.1], dtype=jnp.float32),
        "b": jnp.array([0.05], dtype=jnp.float32),
    }

    opt_pairs = [
        (sgd(0.1), optax.sgd(0.1)),
        (
            sgd(
                constant_schedule(0.1),
                momentum=0.9,
                nesterov=False,
                accumulator_dtype="float32",
            ),
            optax.sgd(
                optax.constant_schedule(0.1),
                momentum=0.9,
                nesterov=False,
                accumulator_dtype=jnp.float32,
            ),
        ),
        (
            sgd(0.1, momentum=0.9, nesterov=True),
            optax.sgd(0.1, momentum=0.9, nesterov=True),
        ),
        (adagrad(0.1), optax.adagrad(0.1)),
        (adagrad(constant_schedule(0.1)), optax.adagrad(optax.constant_schedule(0.1))),
        (adam(0.1, mu_dtype="float32"), optax.adam(0.1, mu_dtype=jnp.float32)),
        (
            adam(constant_schedule(0.1), nesterov=True),
            optax.adam(optax.constant_schedule(0.1), nesterov=True),
        ),
        (
            adamw(0.1, mu_dtype="float32", mask=_true_mask),
            optax.adamw(0.1, mu_dtype=jnp.float32, mask={"w": True, "b": True}),
        ),
        (
            adamw(
                constant_schedule(0.1),
                mask={"w": True, "b": False, "n": False},
                nesterov=True,
            ),
            optax.adamw(
                optax.constant_schedule(0.1),
                mask={"w": True, "b": False},
                nesterov=True,
            ),
        ),
        (lamb(0.1), optax.lamb(0.1)),
        (
            lamb(constant_schedule(0.1), mask=False),
            optax.lamb(optax.constant_schedule(0.1), mask=False),
        ),
        (
            lars(0.1, weight_decay=0.01, weight_decay_mask=True, trust_ratio_mask=None),
            optax.lars(
                0.1, weight_decay=0.01, weight_decay_mask=True, trust_ratio_mask=True
            ),
        ),
        (
            lars(
                constant_schedule(0.1),
                weight_decay_mask=False,
                trust_ratio_mask=False,
                nesterov=True,
            ),
            optax.lars(
                optax.constant_schedule(0.1),
                weight_decay_mask=False,
                trust_ratio_mask=False,
                nesterov=True,
            ),
        ),
        (lion(0.1, mu_dtype="float32"), optax.lion(0.1, mu_dtype=jnp.float32)),
        (
            lion(constant_schedule(0.1), mask={"w": True, "b": False, "n": False}),
            optax.lion(optax.constant_schedule(0.1), mask={"w": True, "b": False}),
        ),
        (adabelief(0.1, nesterov=False), optax.adabelief(0.1, nesterov=False)),
        (
            adabelief(constant_schedule(0.1), nesterov=True),
            optax.adabelief(optax.constant_schedule(0.1), nesterov=True),
        ),
        (fromage(0.1), optax.fromage(0.1)),
        (fromage(constant_schedule(0.1)), optax.fromage(0.1)),
        (novograd(0.1, weight_decay=0.01), optax.novograd(0.1, weight_decay=0.01)),
        (
            novograd(constant_schedule(0.1), weight_decay=0.0),
            optax.novograd(optax.constant_schedule(0.1), weight_decay=0.0),
        ),
        (
            rmsprop(0.1, centered=False, bias_correction=False, eps_in_sqrt=True),
            optax.rmsprop(0.1, centered=False, bias_correction=False, eps_in_sqrt=True),
        ),
        (
            rmsprop(
                0.1,
                centered=False,
                bias_correction=True,
                eps_in_sqrt=False,
                momentum=0.9,
                nesterov=False,
            ),
            optax.rmsprop(
                0.1,
                centered=False,
                bias_correction=True,
                eps_in_sqrt=False,
                momentum=0.9,
                nesterov=False,
            ),
        ),
        (
            rmsprop(
                0.1,
                centered=True,
                bias_correction=False,
                eps_in_sqrt=True,
                momentum=0.9,
                nesterov=True,
            ),
            optax.rmsprop(
                0.1,
                centered=True,
                bias_correction=False,
                eps_in_sqrt=True,
                momentum=0.9,
                nesterov=True,
            ),
        ),
        (
            rmsprop(
                constant_schedule(0.1),
                initial_scale=0.1,
                centered=True,
                bias_correction=True,
                eps_in_sqrt=False,
            ),
            optax.rmsprop(
                optax.constant_schedule(0.1),
                initial_scale=0.1,
                centered=True,
                bias_correction=True,
                eps_in_sqrt=False,
            ),
        ),
        (yogi(0.1), optax.yogi(0.1)),
        (yogi(constant_schedule(0.1)), optax.yogi(optax.constant_schedule(0.1))),
    ]

    for z_opt, o_opt in opt_pairs:
        state = z_opt.init(params)
        ostate = o_opt.init(j_params)
        u1, state1 = z_opt.update(updates, state, params)
        ou1, ostate1 = o_opt.update(j_updates, ostate, j_params)
        u2, _ = z_opt.update(updates, state1, params)
        ou2, _ = o_opt.update(j_updates, ostate1, j_params)
        assert_allclose_optax(
            get_tree_item(u1, "w"), np.asarray(ou1["w"]), rtol=1e-4, atol=1e-4
        )
        assert_allclose_optax(
            get_tree_item(u2, "w"), np.asarray(ou2["w"]), rtol=1e-4, atol=1e-4
        )
        u_empty, _ = z_opt.update(updates, EmptyState(), params)
        assert_allclose_optax(
            get_tree_item(u_empty, "w"), np.asarray(ou1["w"]), rtol=1e-4, atol=1e-4
        )

    novograd_no_wd = novograd(0.1, weight_decay=0.0)
    u_ng, _ = novograd_no_wd.update(updates, novograd_no_wd.init(params), None)
    o_ng_opt = optax.novograd(0.1, weight_decay=0.0)
    ou_ng, _ = o_ng_opt.update(j_updates, o_ng_opt.init(j_params), j_params)
    assert_allclose_optax(
        get_tree_item(u_ng, "w"), np.asarray(ou_ng["w"]), rtol=1e-4, atol=1e-4
    )

    rms_cent = rmsprop(0.1, centered=True)
    rms_uncent_state = rmsprop(0.1, centered=False).init(params)
    u_cross, _ = rms_cent.update(updates, rms_uncent_state, params)
    o_rms_cent = optax.rmsprop(0.1, centered=True)
    ou_cent, _ = o_rms_cent.update(j_updates, o_rms_cent.init(j_params), j_params)
    assert_allclose_optax(
        get_tree_item(u_cross, "w"), np.asarray(ou_cent["w"]), rtol=1e-4, atol=1e-4
    )

    for req_params_opt in (
        adamw(0.1),
        lamb(0.1),
        lars(0.1),
        lion(0.1),
        fromage(0.1),
        novograd(0.1, weight_decay=0.1),
    ):
        st = req_params_opt.init(params)
        with pytest.raises(ValueError):
            req_params_opt.update(updates, st, None)
