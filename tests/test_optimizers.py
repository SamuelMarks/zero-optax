"""Comprehensive unit tests for `zero_optax.optimizers`, `transform`, `update`, and `clipping`."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import zero_chex
from zero_chex.dataclasses import MappableDataclassMixin
from zero_chex.types import ZeroArray
from zero_flax import nnx
from zero_flax.core import FrozenDict
from zero_flax.types import Dtype as FlaxDtype
from zero_flax.types import ensure_array
from zero_orbax.checkpoint import PyTreeCheckpointer

import zero_optax
from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax._tree import (
    _apply_leaf_fn,
    _coerce_pytree_child,
    _coerce_submask,
    _resolve_dtype,
    _to_raw_array_input,
    apply_mask_tree,
    arange,
    asarray,
    leaf_l2_norm,
    logsumexp,
    ones,
    ones_like,
    squeeze_axes,
    tree_l2_norm,
    tree_leaves,
    tree_map,
    tree_zeros_like,
)
from zero_optax._types import (
    DType,
    MaskTree,
    OptStateProtocol,
    Params,
    PyTree,
    Tensor,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    EmptyState,
    GradientTransformation,
    GradientTransformationExtraArgs,
    OptState,
    ScaleByScheduleState,
    ScaleState,
    _as_count_tensor,
    _OptStateDataclassBase,
    _unwrap_orbax_state_dict,
    identity,
    opt_state_from_state_dict,
    opt_state_to_state_dict,
    set_to_zero,
)
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
from zero_optax.schedules import (
    InjectHyperparamsState,
    InjectStatefulHyperparamsState,
    constant_schedule,
    inject_hyperparams,
    inject_stateful_hyperparams,
    linear_schedule,
)
from zero_optax.transform import (
    ScaleByTrustRatioState,
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
    sample_pair = _SamplePair(left=asarray([3.0]), right=None)
    mapped_tup = tree_map(_double, tup_tree)
    mapped_list = tree_map(_double, list_tree)
    mapped_pair = tree_map(_double, sample_pair)
    co_tup = tree_map(lambda a, b: a, (1.0, 2.0), (3.0, 4.0))
    co_3 = tree_map(lambda a, b, c: a, (1.0,), (2.0,), (3.0,))
    assert isinstance(mapped_tup, tuple)
    assert isinstance(mapped_list, list)
    assert isinstance(mapped_pair, _SamplePair)
    assert len(tree_leaves(sample_pair)) == 2
    assert isinstance(co_tup, tuple)
    assert isinstance(co_3, tuple)
    zero_tup = tree_zeros_like(tup_tree)
    zero_pair = tree_zeros_like(sample_pair)
    assert isinstance(zero_tup, tuple) and zero_tup[1] is None
    assert isinstance(zero_pair, _SamplePair) and zero_pair.right is None
    assert float(leaf_l2_norm(None).item()) == 0.0
    assert float(tree_l2_norm(tup_tree, squared=True).item()) == 1.0

    masked_pair = apply_mask_tree(
        _SamplePair(left=1.0, right=2.0),
        _SamplePair(left=3.0, right=4.0),
        (True, False),
    )
    assert isinstance(masked_pair, _SamplePair) and masked_pair == _SamplePair(1.0, 4.0)

    with pytest.raises(ValueError, match="between 1 and 4 PyTrees"):
        tree_map(lambda a: a, 1.0, 2.0, 3.0, 4.0, 5.0)
    with pytest.raises(ValueError, match="between 1 and 4 PyTrees"):
        _apply_leaf_fn(lambda a: a, 1.0, [2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="Mapping"):
        tree_map(lambda a, b: a, {"w": 1.0}, {"other": 1.0})
    with pytest.raises(ValueError, match="sequence"):
        tree_map(lambda a, b: a, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="leaf"):
        tree_map(lambda a, b: a, 1.0, [1.0])

    assert apply_mask_tree((1.0, 2.0), (3.0, 4.0), (True, False)) == (1.0, 4.0)
    assert apply_mask_tree([1.0, 2.0], [3.0, 4.0], [False, True]) == [3.0, 2.0]
    with pytest.raises(ValueError, match="Mask tree container structure"):
        apply_mask_tree(1.0, 2.0, {"unused": True})

    lse = logsumexp(asarray([[0.0, 0.0]]), axis=-1, keepdims=True)
    assert_allclose_optax(lse, np.array([[np.log(2.0)]]))
    lse_sq = logsumexp(asarray([[0.0, 0.0]]), axis=-1, keepdims=False)
    assert_allclose_optax(lse_sq, np.array([np.log(2.0)]))
    assert_allclose_optax(squeeze_axes(asarray([[5.0]]), None), np.array(5.0))
    assert isinstance(ScaleState(), OptStateProtocol)


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

    def _extra_init(params: Params) -> EmptyState:
        """Initialize empty state for extra-args transformation.

        Args:
            params: Parameters PyTree.

        Returns:
            `EmptyState()`.
        """
        del params
        return EmptyState()

    def _extra_update(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: zero_optax.base.ExtraArgValue,
    ) -> tuple[Updates, EmptyState]:
        """Pass updates through unchanged in extra-args transformation.

        Args:
            updates: Updates PyTree.
            state: Current state.
            params: Optional parameters PyTree.
            **extra_args: Optional extra keyword arguments.

        Returns:
            Tuple `(updates, EmptyState())`.
        """
        del state, params, extra_args
        return updates, EmptyState()

    extra_tx: GradientTransformationExtraArgs[EmptyState] = (
        GradientTransformationExtraArgs(_extra_init, _extra_update)
    )
    init_unpacked, update_unpacked = extra_tx
    assert callable(init_unpacked) and callable(update_unpacked)
    with pytest.raises(TypeError, match="GradientTransformationExtraArgs requires"):
        GradientTransformationExtraArgs(_extra_update)
    chained = chain(sgd(0.1), sgd(0.2), extra_tx)
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
        {"w": asarray([10.0, 0.0]), "m": asarray([[10.0, 0.0]]), "n": None},
        agc_tx.init(params),
        {"w": asarray([1.0, 0.0]), "m": asarray([[1.0, 0.0]]), "n": None},
    )
    assert_allclose_optax(get_tree_item(u_agc, "w"), np.array([0.01, 0.0]))
    agc_axis_tx = adaptive_grad_clip(0.01, axis=-1)
    u_agc_axis, _ = agc_axis_tx.update(
        {"w": asarray([[10.0, 0.0]])},
        agc_axis_tx.init({"w": asarray([[1.0, 0.0]])}),
        {"w": asarray([[1.0, 0.0]])},
    )
    assert_allclose_optax(get_tree_item(u_agc_axis, "w"), np.array([[0.01, 0.0]]))
    with pytest.raises(ValueError, match="adaptive_grad_clip"):
        agc_tx.update(updates, agc_tx.init(params), None)


def test_transform_primitives() -> None:
    """Test all `scale_by_*`, `trace`, and `add_decayed_weights` primitives against `optax`."""
    params: dict[str, PyTree] = {"w": asarray([1.0, 2.0]), "n": None}
    updates: dict[str, PyTree] = {"w": asarray([0.2, -0.4]), "n": None}
    j_params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    j_updates = {"w": jnp.array([0.2, -0.4], dtype=jnp.float32)}

    z_sched = linear_schedule(0.1, 0.01, 10)
    o_sched = optax.linear_schedule(0.1, 0.01, 10)
    z_neg_sched = linear_schedule(-0.1, -0.01, 10)
    o_neg_sched = optax.linear_schedule(-0.1, -0.01, 10)

    tx_pairs: list[
        tuple[GradientTransformation[OptState], optax.GradientTransformation]
    ] = [
        (scale(-0.1), optax.scale(-0.1)),
        (
            scale_by_schedule(z_neg_sched),
            optax.scale_by_schedule(o_neg_sched),
        ),
        (scale_by_learning_rate(0.1), optax.scale_by_learning_rate(0.1)),
        (
            scale_by_learning_rate(z_sched, flip_sign=False),
            optax.scale_by_learning_rate(o_sched, flip_sign=False),
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
        if not isinstance(st, (ScaleState, ScaleByTrustRatioState)):
            with pytest.raises(TypeError):
                z_tx.update(updates, EmptyState(), params)

    ng_wd = scale_by_novograd(weight_decay=0.1)
    with pytest.raises(ValueError, match="scale_by_novograd"):
        ng_wd.update(updates, ng_wd.init(params), None)
    ng_zero = scale_by_novograd(weight_decay=0.0)
    u_ng_none, _ = ng_zero.update(updates, ng_zero.init(params), None)
    ou_ng_none, _ = optax.scale_by_novograd(weight_decay=0.0).update(
        j_updates, optax.scale_by_novograd(weight_decay=0.0).init(j_params), j_params
    )
    assert_allclose_optax(
        get_tree_item(u_ng_none, "w"), np.asarray(ou_ng_none["w"]), rtol=1e-4
    )
    aw_nomask = add_decayed_weights(0.01, mask=None)
    u_aw_nomask, _ = aw_nomask.update(updates, aw_nomask.init(params), params)
    assert_allclose_optax(
        get_tree_item(u_aw_nomask, "w"), np.array([0.21, -0.38]), rtol=1e-5
    )
    lr_none_tx = scale_by_learning_rate(None)
    u_lr_none, _ = lr_none_tx.update(updates, lr_none_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_lr_none, "w"), np.array([0.2, -0.4]))
    for lion_mode in ("smooth", "refined"):
        lion_m_tx = scale_by_lion(mode=lion_mode, smooth_beta=2.0)
        u_lm, _ = lion_m_tx.update(updates, lion_m_tx.init(params), params)
        assert get_tree_item(u_lm, "w").shape == (2,)
    with pytest.raises(ValueError, match="scale_by_trust_ratio"):
        scale_by_trust_ratio().update(updates, ScaleByTrustRatioState(), None)
    with pytest.raises(ValueError, match="add_decayed_weights"):
        add_decayed_weights(0.1).update(
            updates, add_decayed_weights(0.1).init(params), None
        )


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

    z_lr = linear_schedule(0.1, 0.02, 10)
    o_lr = optax.linear_schedule(0.1, 0.02, 10)

    opt_pairs: list[
        tuple[GradientTransformation[OptState], optax.GradientTransformation]
    ] = [
        (sgd(0.1), optax.sgd(0.1)),
        (
            sgd(
                z_lr,
                momentum=0.9,
                nesterov=False,
                accumulator_dtype="float32",
            ),
            optax.sgd(
                o_lr,
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
        (adagrad(z_lr), optax.adagrad(o_lr)),
        (adam(0.1, mu_dtype="float32"), optax.adam(0.1, mu_dtype=jnp.float32)),
        (
            adam(z_lr, nesterov=True),
            optax.adam(o_lr, nesterov=True),
        ),
        (
            adamw(0.1, mu_dtype="float32", mask=_true_mask),
            optax.adamw(0.1, mu_dtype=jnp.float32, mask={"w": True, "b": True}),
        ),
        (
            adamw(
                z_lr,
                mask={"w": True, "b": False, "n": False},
                nesterov=True,
            ),
            optax.adamw(
                o_lr,
                mask={"w": True, "b": False},
                nesterov=True,
            ),
        ),
        (lamb(0.1), optax.lamb(0.1)),
        (
            lamb(z_lr, mask=False),
            optax.lamb(o_lr, mask=False),
        ),
        (
            lars(0.1, weight_decay=0.01, weight_decay_mask=True, trust_ratio_mask=None),
            optax.lars(
                0.1, weight_decay=0.01, weight_decay_mask=True, trust_ratio_mask=True
            ),
        ),
        (
            lars(
                z_lr,
                weight_decay_mask=False,
                trust_ratio_mask=False,
                nesterov=True,
            ),
            optax.lars(
                o_lr,
                weight_decay_mask=False,
                trust_ratio_mask=False,
                nesterov=True,
            ),
        ),
        (lion(0.1, mu_dtype="float32"), optax.lion(0.1, mu_dtype=jnp.float32)),
        (
            lion(z_lr, mask={"w": True, "b": False, "n": False}),
            optax.lion(o_lr, mask={"w": True, "b": False}),
        ),
        (adabelief(0.1, nesterov=False), optax.adabelief(0.1, nesterov=False)),
        (
            adabelief(z_lr, nesterov=True),
            optax.adabelief(o_lr, nesterov=True),
        ),
        (fromage(0.1), optax.fromage(0.1)),
        (novograd(0.1, weight_decay=0.01), optax.novograd(0.1, weight_decay=0.01)),
        (
            novograd(z_lr, weight_decay=0.0),
            optax.novograd(o_lr, weight_decay=0.0),
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
                z_lr,
                initial_scale=0.1,
                centered=True,
                bias_correction=True,
                eps_in_sqrt=False,
            ),
            optax.rmsprop(
                o_lr,
                initial_scale=0.1,
                centered=True,
                bias_correction=True,
                eps_in_sqrt=False,
            ),
        ),
        (yogi(0.1), optax.yogi(0.1)),
        (yogi(z_lr), optax.yogi(o_lr)),
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
        with pytest.raises(TypeError):
            z_opt.update(updates, EmptyState(), params)

    fromage_sched_opt = fromage(z_lr)
    f_state = fromage_sched_opt.init(params)
    f_u1, f_state1 = fromage_sched_opt.update(updates, f_state, params)
    o_f0 = optax.fromage(float(o_lr(0)))
    of_u1, _ = o_f0.update(j_updates, o_f0.init(j_params), j_params)
    assert_allclose_optax(
        get_tree_item(f_u1, "w"), np.asarray(of_u1["w"]), rtol=1e-4, atol=1e-4
    )
    f_u2, _ = fromage_sched_opt.update(updates, f_state1, params)
    o_f1 = optax.fromage(float(o_lr(1)))
    of_u2, _ = o_f1.update(j_updates, o_f1.init(j_params), j_params)
    assert_allclose_optax(
        get_tree_item(f_u2, "w"), np.asarray(of_u2["w"]), rtol=1e-4, atol=1e-4
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
    with pytest.raises(TypeError):
        rms_cent.update(updates, rms_uncent_state, params)

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


@zero_chex.dataclass
class _ChexParamNode(MappableDataclassMixin):
    """Mappable dataclass container for testing `zero_chex` PyTree support in `zero_optax`.

    Attributes:
        w: Weight tensor leaf.
        b: Bias tensor leaf.
    """

    w: Tensor
    b: Tensor

    def __getitem__(self, key: str) -> PyTree:
        """Retrieve field `key` as a `zero_optax` `PyTree`.

        Args:
            key: Field name (`'w'` or `'b'`).

        Returns:
            Field tensor value.
        """
        if key == "w":
            return self.w
        if key == "b":
            return self.b
        raise KeyError(key)

    def __chex_field_values__(self) -> tuple[PyTree, ...]:
        """Return field values tuple `(self.w, self.b)`.

        Returns:
            Tuple `(self.w, self.b)`.
        """
        return (self.w, self.b)


def test_ecosystem_integration_flax_chex_orbax_jax(tmp_path: Path) -> None:
    """Test `zero_optax` directly on `zero-flax` models, `zero-chex` trees, `zero-jax` arrays, and `zero-orbax` checkpoints.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # 1. ArrayLikeProtocol, ToListArrayProtocol, DTypeLikeProtocol, and VariableLeafProtocol in asarray
    z_arr = ensure_array([[1.0, 2.0], [3.0, 4.0]])
    t_from_zjax = asarray(z_arr)
    assert t_from_zjax.shape == (2, 2)
    assert asarray([z_arr, z_arr]).shape == (2, 2, 2)
    assert asarray(z_arr, dtype=FlaxDtype("float32")).dtype == DType.float32

    chex_arr = ZeroArray([1.0, 2.0, 3.0, 4.0], shape=(2, 2), dtype=DType.float32)
    t_from_chex = asarray(chex_arr)
    assert t_from_chex.shape == (2, 2)
    assert asarray(chex_arr, dtype="float32").shape == (2, 2)

    param_leaf = nnx.Param(z_arr, tag="trainable")
    assert asarray(param_leaf).shape == (2, 2)
    assert leaf_l2_norm(param_leaf).shape == ()

    # 2. Initialize adamw, sgd, lion, and rmsprop directly on zero_flax.nnx.Linear(4, 2) state
    model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
    params_state = model.state().split(nnx.Param)[0]
    assert isinstance(params_state, nnx.State)
    assert isinstance(params_state["kernel"], nnx.Param)
    assert isinstance(params_state["bias"], nnx.Param)

    grads_state = nnx.State(
        {
            "kernel": nnx.Param(ensure_array([[0.1, 0.1]] * 4)),
            "bias": nnx.Param(ensure_array([0.05, 0.05])),
        }
    )

    for tx in (
        adamw(
            1e-3, mask=nnx.State({"kernel": nnx.Param(True), "bias": nnx.Param(False)})
        ),
        sgd(1e-2, momentum=0.9),
        lion(1e-3),
        rmsprop(1e-3),
        rmsprop(1e-3, bias_correction=True),
        rmsprop(1e-3, centered=True, bias_correction=True),
        adabelief(1e-3),
        novograd(1e-3),
        fromage(1e-3),
    ):
        opt_state = tx.init(params_state)
        zero_chex.assert_trees_all_equal_structs(opt_state, tx.init(params_state))
        zero_chex.assert_trees_all_close(opt_state, tx.init(params_state))

        step_updates, next_opt_state = tx.update(grads_state, opt_state, params_state)
        assert isinstance(step_updates, nnx.State)
        new_params_state = apply_updates(params_state, step_updates)
        assert isinstance(new_params_state, nnx.State)
        assert isinstance(new_params_state["kernel"], nnx.Param)
        assert isinstance(new_params_state["bias"], nnx.Param)
        zero_chex.assert_trees_all_equal_structs(params_state, new_params_state)

        # Round-trip state dictionary serialization for every optimizer state
        s_dict = opt_state_to_state_dict(next_opt_state)
        restored_st = opt_state_from_state_dict(opt_state, s_dict)
        zero_chex.assert_trees_all_equal_structs(next_opt_state, restored_st)
        zero_chex.assert_trees_all_close(next_opt_state, restored_st)

    # Update the nnx.Module with updated State and run a forward pass
    adamw_tx = adamw(1e-2)
    adamw_state = adamw_tx.init(params_state)
    u_step, adamw_state = adamw_tx.update(grads_state, adamw_state, params_state)
    updated_model_state = apply_updates(params_state, u_step)
    assert isinstance(updated_model_state, nnx.State)
    model.update(updated_model_state)
    out = model(ensure_array([[1.0, 1.0, 1.0, 1.0]]))
    assert out.shape == (1, 2)

    # 3. FrozenDict and zero_chex.dataclass support in tree_map, tree_leaves, and apply_mask_tree
    fdict = FrozenDict(
        {"w": asarray([1.0, 2.0]), "b": FrozenDict({"sub": asarray([0.5])})}
    )
    mapped_fdict = tree_map(lambda x: asarray(x) * 2.0, fdict)
    assert isinstance(mapped_fdict, FrozenDict)
    assert len(tree_leaves(fdict)) == 2
    masked_fdict = apply_mask_tree(
        mapped_fdict,
        fdict,
        FrozenDict({"w": True, "b": {"sub": False}}),
    )
    assert isinstance(masked_fdict, FrozenDict)
    assert _coerce_submask((True, False)) == (True, False)
    list_mask: MaskTree = [True, False]
    assert _coerce_submask(list_mask) == [True, False]
    assert _coerce_submask({"a": True}) == {"a": True}
    assert _coerce_submask(1) is True

    dc1 = _ChexParamNode(w=asarray([1.0, 2.0]), b=asarray([3.0]))
    dc2 = _ChexParamNode(w=asarray([0.5, 0.5]), b=asarray([1.0]))
    dc_sum = tree_map(lambda a, b: asarray(a) + asarray(b), dc1, dc2)
    assert isinstance(dc_sum, _ChexParamNode)
    assert len(tree_leaves(dc_sum)) == 2
    assert opt_state_to_state_dict(dc_sum) is not None
    with pytest.raises(ValueError, match="expected MappableDataclass"):
        tree_map(lambda a, b: a, dc1, {"w": asarray([1.0, 2.0]), "b": asarray([3.0])})

    # 4. Save and restore chained OptState via zero_orbax.checkpoint.PyTreeCheckpointer
    chained_tx = chain(
        clip_by_global_norm(1.0),
        add_decayed_weights(linear_schedule(0.01, 0.001, 10)),
        adamw(1e-3),
    )
    c_state0 = chained_tx.init(params_state)
    _u_before, c_state1 = chained_tx.update(grads_state, c_state0, params_state)

    ckpt_dir = tmp_path / "opt_ckpt"
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(ckpt_dir, opt_state_to_state_dict(c_state1))
    raw_restored = checkpointer.restore(ckpt_dir)
    c_state1_restored = opt_state_from_state_dict(c_state0, raw_restored)
    zero_chex.assert_trees_all_equal_structs(c_state1, c_state1_restored)
    zero_chex.assert_trees_all_close(c_state1, c_state1_restored)

    u_expected, c_state2_expected = chained_tx.update(
        grads_state, c_state1, params_state
    )
    u_restored, c_state2_restored = chained_tx.update(
        grads_state, c_state1_restored, params_state
    )
    zero_chex.assert_trees_all_close(u_expected, u_restored)
    zero_chex.assert_trees_all_close(c_state2_expected, c_state2_restored)

    # 5. Test _OptStateDataclassBase, InjectHyperparamsState, InjectStatefulHyperparamsState, and edge cases
    base_inst = _OptStateDataclassBase()
    assert base_inst._asdict() == {}
    assert base_inst.__chex_mappable_fields__() == ()
    assert base_inst.__chex_field_values__() == ()
    assert list(base_inst) == []
    assert len(base_inst) == 0

    sched_st = ScaleByScheduleState(count=asarray(3))
    assert sched_st["count"] is not None
    assert list(sched_st) == ["count"]
    assert len(sched_st) == 1
    assert ScaleByScheduleState._make([5]).count.shape == ()
    assert _as_count_tensor(None).shape == ()

    inj_factory = inject_hyperparams(adam)
    assert inj_factory is not None
    inj_tx = inj_factory(linear_schedule(0.1, 0.01, 10))
    inj_st = inj_tx.init(params_state)
    _, inj_st_next = inj_tx.update(grads_state, inj_st, params_state)
    inj_dict = opt_state_to_state_dict(inj_st_next)
    inj_restored = opt_state_from_state_dict(inj_st, inj_dict)
    assert isinstance(inj_restored, InjectHyperparamsState)
    assert InjectHyperparamsState._make([1, None]).hyperparams == {}
    assert (
        "lr"
        in InjectHyperparamsState._make([1, {"lr": ensure_array(0.05)}]).hyperparams
    )

    stateful_factory = inject_stateful_hyperparams(sgd)
    assert stateful_factory is not None
    st_inj_tx = stateful_factory(constant_schedule(0.05))
    st_inj_state = st_inj_tx.init(params_state)
    _, st_inj_next = st_inj_tx.update(grads_state, st_inj_state, params_state)
    st_inj_dict = opt_state_to_state_dict(st_inj_next)
    st_inj_restored = opt_state_from_state_dict(st_inj_state, st_inj_dict)
    assert isinstance(st_inj_restored, InjectStatefulHyperparamsState)
    assert InjectStatefulHyperparamsState._make([1, {}, None]).hyperparams_states == {}

    # Tuple/list/Variable/FrozenDict/error branches in opt_state_to_state_dict and opt_state_from_state_dict
    list_state: OptState = [ScaleByScheduleState(count=asarray(1))]
    list_dict = opt_state_to_state_dict(list_state)
    assert isinstance(opt_state_from_state_dict(list_state, list_dict), list)
    zero_chex.assert_trees_all_equal_structs(list_state, list_state)
    assert isinstance(
        opt_state_from_state_dict(sched_st, (asarray(2),)), ScaleByScheduleState
    )
    assert isinstance(opt_state_from_state_dict(fdict, fdict), FrozenDict)
    assert isinstance(opt_state_from_state_dict(param_leaf, param_leaf), nnx.Param)
    assert opt_state_to_state_dict(z_arr) is not None
    assert zero_optax.MaskedNode().__orbax_state_type__ == "MaskedNode"
    assert sched_st.__orbax_state_type__ == "ScaleByScheduleState"
    mn_dict = opt_state_to_state_dict(zero_optax.MaskedNode())
    assert mn_dict == {"__orbax_state_type__": "MaskedNode"}
    assert isinstance(
        opt_state_from_state_dict(zero_optax.MaskedNode(), mn_dict),
        zero_optax.MaskedNode,
    )
    assert isinstance(
        _unwrap_orbax_state_dict(zero_optax.MaskedNode()),
        zero_optax.MaskedNode,
    )
    assert isinstance(_unwrap_orbax_state_dict(mn_dict), zero_optax.MaskedNode)
    assert isinstance(_unwrap_orbax_state_dict(sched_st), dict)
    adan_st = zero_optax.ScaleByAdanState(
        m=asarray(1.0),
        v=asarray(2.0),
        n=asarray(3.0),
        g=asarray(4.0),
        t=asarray(5),
    )
    assert adan_st._asdict()["t"] is not None
    assert isinstance(
        zero_optax.ScaleByAdanState._make([1.0, 2.0, 3.0, 4.0, 5]),
        zero_optax.ScaleByAdanState,
    )
    assert isinstance(
        ScaleByScheduleState.tree_unflatten(None, [asarray(3)]),
        ScaleByScheduleState,
    )

    class _CustomVarLeaf:
        """Minimal `VariableLeafProtocol` test implementation outside `nnx.Variable`."""

        def __init__(self, value: PyTree | None = None) -> None:
            """Initialize `_CustomVarLeaf` with `value`.

            Args:
                value: Stored leaf value.
            """
            self._v: Tensor = asarray(0.0) if value is None else asarray(1.0)

        @property
        def value(self) -> Tensor:
            """Return stored value.

            Returns:
                Stored `Tensor` value.
            """
            return self._v

        def copy(self) -> _CustomVarLeaf:
            """Return copy of `_CustomVarLeaf`.

            Returns:
                Copied `_CustomVarLeaf`.
            """
            return _CustomVarLeaf(self._v)

        def replace(self) -> _CustomVarLeaf:
            """Return replaced `_CustomVarLeaf`.

            Returns:
                New `_CustomVarLeaf`.
            """
            return _CustomVarLeaf(self._v)

    assert isinstance(
        tree_map(lambda x: asarray(x) + 1.0, _CustomVarLeaf(asarray(1.0))),
        _CustomVarLeaf,
    )
    assert zero_optax.ScaleByAddNoiseState is zero_optax.AddNoiseState
    with pytest.raises(TypeError, match="Unsupported ArrayLike"):
        _to_raw_array_input(None)
    assert _resolve_dtype(int) == DType.int32
    assert _coerce_pytree_child([True, False]) == [True, False]
    assert _coerce_pytree_child("not_a_pytree") is None
    assert _unwrap_orbax_state_dict([1, 2]) == [1, 2]
    assert _unwrap_orbax_state_dict("not_a_pytree") is None
    assert opt_state_to_state_dict(42) == 42
    assert opt_state_from_state_dict(42, 99) == 99
    with pytest.raises(TypeError, match="Cannot restore"):
        opt_state_from_state_dict(sched_st, 123)


def test_phase4_transforms_clipping_and_linesearch() -> None:
    """Test stateless, zero_nans, centralize, keep_params_nonnegative, block rms, noise, ema, amsgrad, and linesearch."""
    params: dict[str, PyTree] = {
        "w": asarray([1.0, 2.0]),
        "m": asarray([[1.0, 3.0], [2.0, 4.0]]),
        "s": asarray(1.5),
        "n": None,
    }
    updates: dict[str, PyTree] = {
        "w": asarray([-2.0, 0.5]),
        "m": asarray([[1.0, 3.0], [2.0, 4.0]]),
        "s": asarray(0.5),
        "n": None,
    }

    st_tx = zero_optax.stateless(lambda u, p: u)
    u_st, _ = st_tx.update(updates, st_tx.init(params), params)
    assert u_st is updates

    zn_tx = zero_optax.zero_nans()
    zn_state = zn_tx.init(params)
    nan_updates: dict[str, PyTree] = {
        "w": asarray([float("nan"), 1.0]),
        "m": asarray([[1.0, 2.0], [3.0, 4.0]]),
        "s": asarray(float("nan")),
        "n": None,
    }
    u_zn, zn_state2 = zn_tx.update(nan_updates, zn_state, params)
    assert_allclose_optax(get_tree_item(u_zn, "w"), np.array([0.0, 1.0]))
    assert bool(get_tree_item(zn_state2.found_nan, "w").item()) is True

    cent_tx = zero_optax.centralize()
    u_cent, _ = cent_tx.update(updates, cent_tx.init(params), params)
    assert_allclose_optax(
        get_tree_item(u_cent, "m"), np.array([[-1.0, 1.0], [-1.0, 1.0]])
    )

    kpn_tx = zero_optax.keep_params_nonnegative()
    u_kpn, _ = kpn_tx.update(updates, kpn_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_kpn, "w"), np.array([-1.0, 0.5]))
    with pytest.raises(ValueError, match="keep_params_nonnegative"):
        kpn_tx.update(updates, kpn_tx.init(params), None)

    cbr_tx = zero_optax.clip_by_block_rms(0.5)
    u_cbr, _ = cbr_tx.update(updates, cbr_tx.init(params), params)
    assert get_tree_item(u_cbr, "w").shape == (2,)

    with pytest.raises(ValueError, match="l2_norm_clip must be >= 0"):
        zero_optax.per_example_global_norm_clip(updates, -1.0)
    empty_pe, empty_cnt = zero_optax.per_example_global_norm_clip({"n": None}, 1.0)
    assert int(empty_cnt.item()) == 0 and empty_pe == {"n": None}
    batched_g: dict[str, PyTree] = {
        "w": asarray([[3.0, 4.0], [0.1, 0.2], [0.0, 0.0]]),
        "n": None,
    }
    sum_pe, cnt_pe = zero_optax.per_example_global_norm_clip(batched_g, 1.0)
    assert int(cnt_pe.item()) == 1
    assert_allclose_optax(get_tree_item(sum_pe, "w"), np.array([0.7, 1.0]))
    _, cnt_pe0 = zero_optax.per_example_global_norm_clip(
        {"w": asarray([[0.0, 0.0]])}, 0.0
    )
    assert int(cnt_pe0.item()) == 0

    with pytest.raises(ValueError, match="Provide only one"):
        zero_optax.add_noise(0.1, 0.5, key=1, seed=2)
    an_tx1 = zero_optax.add_noise(0.1, 0.5)
    an_tx2 = zero_optax.add_noise(0.1, 0.5, key=7)
    an_tx3 = zero_optax.add_noise(0.1, 0.5, seed=9)
    for tx_n in (an_tx1, an_tx2, an_tx3):
        st_n = tx_n.init(params)
        assert st_n._asdict()["count"] is not None
        u_n, _ = tx_n.update(updates, st_n, params)
        assert get_tree_item(u_n, "w").shape == (2,)
    with pytest.raises(TypeError, match="AddNoiseState"):
        an_tx1.update(updates, EmptyState(), params)

    for deb in (True, False):
        ema_tx = zero_optax.ema(0.9, debias=deb, accumulator_dtype="float32")
        st_e = ema_tx.init(params)
        assert st_e._asdict()["count"] is not None
        u_e, _ = ema_tx.update(updates, st_e, params)
        assert get_tree_item(u_e, "w").shape == (2,)
    ema_tx_nodt = zero_optax.ema(1.0, debias=True, accumulator_dtype=None)
    ema_tx_nodt.update(updates, ema_tx_nodt.init(params), params)
    with pytest.raises(TypeError, match="EmaState"):
        ema_tx_nodt.update(updates, EmptyState(), params)

    for bc_flag, mdt in ((True, "float32"), (False, None)):
        ams_tx = zero_optax.scale_by_amsgrad(
            b1=1.0 if not bc_flag else 0.9,
            b2=1.0 if not bc_flag else 0.999,
            mu_dtype=mdt,
            bias_correction_mu=bc_flag,
            bias_correction_nu=bc_flag,
        )
        st_a = ams_tx.init(params)
        assert st_a._asdict()["count"] is not None
        u_a, _ = ams_tx.update(updates, st_a, params)
        assert get_tree_item(u_a, "w").shape == (2,)
    ams_b1_one = zero_optax.scale_by_amsgrad(b1=1.0, b2=1.0)
    ams_b1_one.update(updates, ams_b1_one.init(params), params)
    with pytest.raises(TypeError, match="ScaleByAmsgradState"):
        ams_b1_one.update(updates, EmptyState(), params)

    pbn_tx = zero_optax.scale_by_param_block_norm(0.01)
    u_pbn, _ = pbn_tx.update(updates, pbn_tx.init(params), params)
    assert get_tree_item(u_pbn, "w").shape == (2,)
    with pytest.raises(ValueError, match="scale_by_param_block_norm"):
        pbn_tx.update(updates, pbn_tx.init(params), None)

    pbr_tx = zero_optax.scale_by_param_block_rms(0.01)
    u_pbr, _ = pbr_tx.update(updates, pbr_tx.init(params), params)
    assert get_tree_item(u_pbr, "w").shape == (2,)
    with pytest.raises(ValueError, match="scale_by_param_block_rms"):
        pbr_tx.update(updates, pbr_tx.init(params), None)

    def _quad_fn(p: Params, scale_f: float = 1.0) -> Tensor:
        """Compute quadratic objective `scale_f * sum(w^2 + s^2)`.

        Args:
            p: Parameters dict.
            scale_f: Multiplier.

        Returns:
            Scalar objective Tensor.
        """
        if not isinstance(p, dict):
            return asarray(0.0)
        w_t = asarray(p["w"])
        s_t = asarray(p["s"])
        return scale_f * (zero_optax._tree.reduce_sum(w_t * w_t) + s_t * s_t)

    def _var_kw_fn(p: Params, **kw: PyTree) -> Tensor:
        """Quadratic objective accepting `**kw`.

        Args:
            p: Parameters dict.
            **kw: Extra keyword arguments.

        Returns:
            Scalar objective Tensor.
        """
        del kw
        return _quad_fn(p)

    bt_tx = zero_optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=5, store_grad=True
    )
    bt_st = bt_tx.init(params)
    vg_helper = zero_optax.value_and_grad_from_state(_quad_fn)
    val0, grad0 = vg_helper(params, state=bt_st)
    from zero_optax._linesearch import _call_value_fn

    assert _call_value_fn(lambda p, x: 1.0, params, (1.0,)) == 1.0
    assert _call_value_fn(lambda p, x, y: 2.0, params, (1.0, 2.0)) == 2.0
    with pytest.raises(ValueError, match="Unsupported value_fn"):
        _call_value_fn(_quad_fn, params, (1.0, 2.0, 3.0))
    descent = zero_optax.tree_utils.tree_scalar_mul(-1.0, grad0)
    u_bt, bt_st2 = bt_tx.update(
        descent, bt_st, params, value=val0, grad=grad0, value_fn=_quad_fn, scale_f=1.0
    )
    assert get_tree_item(u_bt, "w").shape == (2,)
    val1, grad1 = vg_helper(params, state=bt_st2)
    assert float(val1.item()) < float(val0.item())
    assert grad1 is not None

    bt_no_sg = zero_optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=2, store_grad=False
    )
    bt_no_st = bt_no_sg.init(params)
    bt_no_sg.update(grad0, bt_no_st, params, value_fn=_var_kw_fn, extra_param=1.0)
    with pytest.raises(ValueError, match="Value or gradient not found"):
        vg_helper(params, state=bt_no_st)
    with pytest.raises(ValueError, match="params"):
        bt_tx.update(descent, bt_st, None, value_fn=_quad_fn)
    with pytest.raises(TypeError, match="ScaleByBacktrackingLinesearchState"):
        bt_tx.update(descent, EmptyState(), params, value_fn=_quad_fn)
    with pytest.raises(ValueError, match="value_fn"):
        bt_tx.update(descent, bt_st, params)

    for strat, mlr in (("keep", None), ("one", 0.5)):
        zm_tx = zero_optax.scale_by_zoom_linesearch(
            max_linesearch_steps=4,
            max_learning_rate=mlr,
            initial_guess_strategy=strat,
        )
        zm_st = zm_tx.init(params)
        u_zm, _ = zm_tx.update(descent, zm_st, params, value_fn=_quad_fn)
        assert get_tree_item(u_zm, "w").shape == (2,)
        zm_tx.update(grad0, zm_st, params, value=val0, grad=grad0, value_fn=_quad_fn)

    zm_expand = zero_optax.scale_by_zoom_linesearch(
        max_linesearch_steps=3,
        max_learning_rate=10.0,
        curv_rtol=0.0,
        initial_guess_strategy="one",
    )
    zm_expand.update(
        zero_optax.tree_utils.tree_scalar_mul(0.01, descent),
        zm_expand.init(params),
        params,
        value_fn=_quad_fn,
    )
    zm_expand_nomax = zero_optax.scale_by_zoom_linesearch(
        max_linesearch_steps=3,
        max_learning_rate=None,
        curv_rtol=0.0,
        initial_guess_strategy="one",
    )
    zm_expand_nomax.update(
        zero_optax.tree_utils.tree_scalar_mul(0.01, descent),
        zm_expand_nomax.init(params),
        params,
        value_fn=_quad_fn,
    )
    zm_pos_slope = zero_optax.scale_by_zoom_linesearch(
        max_linesearch_steps=3,
        slope_rtol=0.0,
        curv_rtol=0.0,
    )
    zm_pos_slope.update(
        zero_optax.tree_utils.tree_scalar_mul(0.8, descent),
        zm_pos_slope.init(params),
        params,
        value_fn=_quad_fn,
    )
    with pytest.raises(ValueError, match="params"):
        zm_expand.update(descent, zm_expand.init(params), None, value_fn=_quad_fn)
    with pytest.raises(TypeError, match="ScaleByZoomLinesearchState"):
        zm_expand.update(descent, EmptyState(), params, value_fn=_quad_fn)
    with pytest.raises(ValueError, match="value_fn"):
        zm_expand.update(descent, zm_expand.init(params), params)


def test_phase5_combinators_and_wrappers() -> None:
    """Test named_chain, masked, freeze, flatten, partition, multi_transform, lookahead, and MultiSteps."""
    params: dict[str, PyTree] = {
        "w": asarray([1.0, 2.0]),
        "b": asarray([0.5]),
        "s": asarray(1.0),
        "n": None,
    }
    updates: dict[str, PyTree] = {
        "w": asarray([0.1, -0.2]),
        "b": asarray([0.05]),
        "s": asarray(0.2),
        "n": None,
    }

    def _ex_init(params: Params) -> EmptyState:
        """Initialize empty state.

        Args:
            params: Parameters PyTree.

        Returns:
            `EmptyState`.
        """
        del params
        return EmptyState()

    def _ex_upd(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: zero_optax.base.ExtraArgValue,
    ) -> tuple[Updates, EmptyState]:
        """Pass updates through unchanged.

        Args:
            updates: Updates PyTree.
            state: State.
            params: Optional parameters.
            **extra_args: Extra keyword args.

        Returns:
            Tuple `(updates, EmptyState())`.
        """
        del state, params, extra_args
        return updates, EmptyState()

    ex_tx: GradientTransformationExtraArgs[EmptyState] = (
        GradientTransformationExtraArgs(_ex_init, _ex_upd)
    )

    with pytest.raises(ValueError, match="unique names"):
        zero_optax.named_chain(("a", sgd(0.1)), ("a", sgd(0.2)))
    nc_tx = zero_optax.named_chain(("step1", sgd(0.1)), ("step2", ex_tx))
    nc_st = nc_tx.init(params)
    assert nc_st["step1"] is not None
    u_nc, nc_st2 = nc_tx.update(updates, nc_st, params, value=1.0)
    assert get_tree_item(u_nc, "w").shape == (2,)
    assert "step2" in nc_st2
    with pytest.raises(TypeError, match="named_chain"):
        nc_tx.update(updates, EmptyState(), params)

    m_tx = zero_optax.masked(sgd(0.1), {"w": True, "b": False, "s": True, "n": False})
    m_st = m_tx.init(params)
    u_m, _ = m_tx.update(updates, m_st, params)
    assert_allclose_optax(get_tree_item(u_m, "b"), np.array([0.05]))
    m_tx_extra = zero_optax.masked(ex_tx, True)
    m_tx_extra.update(updates, m_tx_extra.init(params), None)
    m_tx_none = zero_optax.masked(sgd(0.1), None)
    m_tx_none.update(updates, m_tx_none.init(params), None)
    with pytest.raises(TypeError, match="MaskedState"):
        m_tx.update(updates, EmptyState(), params)

    frz_tx = zero_optax.freeze({"w": True, "b": False, "s": False, "n": False})
    u_frz, _ = frz_tx.update(updates, frz_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_frz, "w"), np.array([0.0, 0.0]))
    assert_allclose_optax(get_tree_item(u_frz, "b"), np.array([0.05]))

    fl_tx = zero_optax.flatten(sgd(0.1))
    fl_st = fl_tx.init(params)
    u_fl, _ = fl_tx.update(updates, fl_st, params)
    assert_allclose_optax(get_tree_item(u_fl, "w"), np.array([-0.01, 0.02]))
    fl_tx.update(updates, fl_st, None)
    fl_tx.update({"n": None}, fl_tx.init({"n": None}), None)
    fl_ex = zero_optax.flatten(ex_tx)
    fl_ex.update(updates, fl_ex.init(params), params)

    part_tx = zero_optax.partition(
        {"w_grp": sgd(0.1), "b_grp": sgd(1.0)},
        {"w": "w_grp", "b": "b_grp", "s": "w_grp", "n": "b_grp"},
    )
    part_st = part_tx.init(params)
    u_part, _ = part_tx.update(updates, part_st, params)
    assert_allclose_optax(get_tree_item(u_part, "w"), np.array([-0.01, 0.02]))
    assert_allclose_optax(get_tree_item(u_part, "b"), np.array([-0.05]))
    with pytest.raises(TypeError, match="PartitionState"):
        part_tx.update(updates, EmptyState(), params)

    mt_tx = zero_optax.multi_transform(
        {"g": sgd(0.1)},
        lambda p: ("g", ["g"]),
    )
    assert mt_tx is not None
    from zero_optax.combine import _build_label_mask

    assert isinstance(_build_label_mask(("g", ["g"]), "g"), tuple)
    assert isinstance(
        _build_label_mask(_ChexParamNode(w=asarray([1.0]), b=asarray([2.0])), "g"),
        dict,
    )
    assert isinstance(
        _build_label_mask(FrozenDict({"w": asarray([1.0])}), "g"),
        dict,
    )

    with pytest.raises(ValueError, match="Synchronization period"):
        zero_optax.lookahead(sgd(0.1), sync_period=0, slow_step_size=0.5)
    la_params = zero_optax.LookaheadParams.init_synced(params)
    for rst in (False, True):
        la_tx = zero_optax.lookahead(
            sgd(0.1), sync_period=2, slow_step_size=0.5, reset_state=rst
        )
        la_st = la_tx.init(la_params)
        assert la_st._asdict()["slow_params"] is not None
        u_la1, la_st1 = la_tx.update(updates, la_st, la_params)
        assert isinstance(u_la1, zero_optax.LookaheadParams)
        u_la2, _ = la_tx.update(updates, la_st1, la_params)
        assert isinstance(u_la2, zero_optax.LookaheadParams)
    la_plain = zero_optax.lookahead(sgd(0.1), sync_period=1, slow_step_size=0.5)
    la_plain_st = la_plain.init(params)
    la_plain.update(updates, la_plain_st, params)
    with pytest.raises(TypeError, match="LookaheadState"):
        la_plain.update(updates, EmptyState(), params)
    with pytest.raises(ValueError, match="lookahead"):
        la_plain.update(updates, la_plain_st, None)

    flag_nf, _ = zero_optax.skip_not_finite(
        {"w": asarray([float("nan"), 1.0]), "n": None}, asarray(0), params
    )
    assert bool(flag_nf.item()) is True
    flag_ok, _ = zero_optax.skip_not_finite(updates, asarray(0), params)
    assert bool(flag_ok.item()) is False
    flag_ld, _ = zero_optax.skip_large_deltas(
        {"w": asarray([1000.0]), "n": None}, asarray(0), params, max_Sqr_norm=10.0
    )
    assert bool(flag_ld.item()) is True

    ms = zero_optax.MultiSteps(sgd(0.1), every_k_schedule=2, use_grad_mean=True)
    assert ms.inner_opt is not None
    ms_st = ms.init(params)
    assert ms_st._asdict()["mini_step"] is not None
    assert bool(ms.has_updated(ms_st).item()) is False
    u_ms1, ms_st1 = ms.update(updates, ms_st, params)
    assert_allclose_optax(get_tree_item(u_ms1, "w"), np.array([0.0, 0.0]))
    assert bool(ms.has_updated(ms_st1).item()) is False
    u_ms2, ms_st2 = ms.update(updates, ms_st1, params)
    assert_allclose_optax(get_tree_item(u_ms2, "w"), np.array([-0.01, 0.02]))
    assert bool(ms.has_updated(ms_st2).item()) is True
    ms_gt = ms.gradient_transformation()
    assert isinstance(ms_gt, GradientTransformationExtraArgs)

    ms_sched = zero_optax.MultiSteps(
        ex_tx,
        every_k_schedule=lambda step: 1,
        use_grad_mean=False,
        should_skip_update_fn=zero_optax.skip_not_finite,
    )
    ms_sched_st = ms_sched.init(params)
    ms_sched.update(updates, ms_sched_st, params)
    ms_sched.update(
        {
            "w": asarray([float("nan"), 1.0]),
            "b": asarray([0.0]),
            "s": asarray(0.0),
            "n": None,
        },
        ms_sched_st,
        params,
    )
    with pytest.raises(TypeError, match=r"MultiSteps\.update"):
        ms.update(updates, EmptyState(), params)
    with pytest.raises(TypeError, match="has_updated"):
        ms.has_updated(EmptyState())


def test_phase6_additional_optimizers() -> None:
    """Test adafactor, adamax, adamaxw, amsgrad, lbfgs, noisy_sgd, ogd, polyak_sgd, radam, rprop, sign_sgd, and sm3."""
    params: dict[str, PyTree] = {
        "w": asarray([1.0, 2.0]),
        "m": asarray([[1.0, 2.0], [3.0, 4.0]]),
        "n": None,
    }
    updates: dict[str, PyTree] = {
        "w": asarray([0.2, -0.4]),
        "m": asarray([[0.1, -0.2], [0.3, -0.1]]),
        "n": None,
    }

    for opt in (
        zero_optax.adamax(0.1),
        zero_optax.adamax(0.1, b1=1.0),
        zero_optax.adamaxw(0.1, weight_decay=0.01),
        zero_optax.amsgrad(0.1),
        zero_optax.noisy_sgd(0.1, key=0),
        zero_optax.sign_sgd(0.1),
        zero_optax.optimistic_gradient_descent(0.1),
        zero_optax.radam(0.1, threshold=5.0, nesterov=False),
        zero_optax.radam(0.1, threshold=0.1, nesterov=True),
        zero_optax.radam(0.1, b1=1.0, b2=1.0, threshold=0.1, nesterov=True),
        zero_optax.rprop(0.1),
        zero_optax.rprop(lambda count: 0.1),
        zero_optax.sm3(0.1, momentum=0.9),
        zero_optax.sm3(0.1, momentum=0.0),
        zero_optax.adafactor(
            learning_rate=None,
            min_dim_size_to_factor=2,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,
            momentum=0.9,
            weight_decay_rate=0.01,
            factored=True,
        ),
        zero_optax.adafactor(
            learning_rate=None,
            min_dim_size_to_factor=2,
            multiply_by_parameter_scale=False,
            clipping_threshold=None,
            momentum=None,
            weight_decay_rate=None,
            factored=False,
        ),
    ):
        st = opt.init(params)
        u1, st1 = opt.update(updates, st, params)
        assert get_tree_item(u1, "w").shape == (2,)
        u2, _ = opt.update(updates, st1, params)
        assert get_tree_item(u2, "w").shape == (2,)

    sm3_b2 = zero_optax.scale_by_sm3(b1=0.9, b2=0.9)
    sm3_b2.update(updates, sm3_b2.init(params), params)

    for state_cls_tx in (
        zero_optax.scale_by_adamax(),
        zero_optax.scale_by_optimistic_gradient(),
        zero_optax.scale_by_radam(),
        zero_optax.scale_by_rprop(0.1),
        zero_optax.scale_by_sm3(),
        zero_optax.scale_by_factored_rms(),
    ):
        st_sub = state_cls_tx.init(params)
        if isinstance(st_sub, zero_optax.base._OptStateDataclassBase):
            assert isinstance(st_sub._asdict(), dict)
        with pytest.raises(TypeError):
            state_cls_tx.update(updates, EmptyState(), params)

    with pytest.raises(ValueError, match="Invalid argument for `variant`"):
        zero_optax.polyak_sgd(variant="bad_variant")
    for var_mode in ("sps", "sps+"):
        psgd = zero_optax.polyak_sgd(variant=var_mode)
        pst = psgd.init(params)
        u_ps, _ = psgd.update(updates, pst, params, value=2.0)
        assert get_tree_item(u_ps, "w").shape == (2,)
        psgd.update(
            {
                "w": asarray([0.0, 0.0]),
                "m": asarray([[0.0, 0.0], [0.0, 0.0]]),
                "n": None,
            },
            pst,
            params,
            value=2.0,
        )
    with pytest.raises(ValueError, match="value"):
        zero_optax.polyak_sgd().update(
            updates, zero_optax.polyak_sgd().init(params), params
        )

    with pytest.raises(ValueError, match="memory_size"):
        zero_optax.scale_by_lbfgs(memory_size=0)
    with pytest.raises(ValueError, match="scale_by_lbfgs"):
        zero_optax.scale_by_lbfgs().update(
            updates, zero_optax.scale_by_lbfgs().init(params), None
        )
    with pytest.raises(TypeError, match="ScaleByLBFGSState"):
        zero_optax.scale_by_lbfgs().update(updates, EmptyState(), params)

    for sc_init, mem_sz in ((True, 1), (False, 2)):
        lbfgs_opt = zero_optax.lbfgs(
            learning_rate=0.1,
            memory_size=mem_sz,
            scale_init_precond=sc_init,
            linesearch=None,
        )
        lb_st = lbfgs_opt.init(params)
        assert isinstance(lb_st[0], zero_optax.ScaleByLBFGSState)
        assert lb_st[0]._asdict()["count"] is not None
        u_lb0, lb_st1 = lbfgs_opt.update(updates, lb_st, params)
        p_next = zero_optax.apply_updates(params, u_lb0)
        g_next: dict[str, PyTree] = {
            "w": asarray([0.1, -0.2]),
            "m": asarray([[0.05, -0.1], [0.15, -0.05]]),
            "n": None,
        }
        u_lb1, lb_st2 = lbfgs_opt.update(g_next, lb_st1, p_next)
        p_next2 = zero_optax.apply_updates(p_next, u_lb1)
        g_next2: dict[str, PyTree] = {
            "w": asarray([0.05, -0.1]),
            "m": asarray([[0.02, -0.05], [0.07, -0.02]]),
            "n": None,
        }
        lbfgs_opt.update(g_next2, lb_st2, p_next2)
        lbfgs_opt.update(g_next2, lb_st2, p_next2)
        lbfgs_opt.update(g_next, lb_st1, params)
    lbfgs_zero = zero_optax.scale_by_lbfgs(scale_init_precond=True)
    u_lzero, st_lzero = lbfgs_zero.update(
        {"w": asarray([0.0, 0.0]), "m": asarray([[0.0, 0.0], [0.0, 0.0]]), "n": None},
        lbfgs_zero.init(params),
        params,
    )
    assert_allclose_optax(get_tree_item(u_lzero, "w"), np.array([0.0, 0.0]))
    assert int(st_lzero.count.item()) == 1

    # Verify _make roundtrip on all _OptStateDataclassBase state classes
    base_dc = zero_optax.base._OptStateDataclassBase._make(())
    assert isinstance(base_dc, zero_optax.base._OptStateDataclassBase)
    dc_states: list[zero_optax.base._OptStateDataclassBase] = [
        zero_optax.add_noise(0.1, 0.5, seed=0).init(params),
        zero_optax.ema(0.9).init(params),
        zero_optax.scale_by_amsgrad().init(params),
        zero_optax.scale_by_adamax().init(params),
        zero_optax.scale_by_factored_rms(min_dim_size_to_factor=2).init(params),
        zero_optax.scale_by_lbfgs().init(params),
        zero_optax.scale_by_optimistic_gradient().init(params),
        zero_optax.scale_by_rprop(0.1).init(params),
        zero_optax.lookahead(sgd(0.1), 2, 0.5).init(params),
        zero_optax.MultiSteps(sgd(0.1), 2).init(params),
        InjectHyperparamsState(
            count=asarray(0),
            hyperparams={"learning_rate": 0.1},
            inner_state=sgd(0.1).init(params),
        ),
        InjectStatefulHyperparamsState(
            count=asarray(0),
            hyperparams={"learning_rate": 0.1},
            hyperparams_states={},
            inner_state=sgd(0.1).init(params),
        ),
    ]
    for dc_st in dc_states:
        rebuilt_st = dc_st._make(list(dc_st._asdict().values()))
        assert type(rebuilt_st) is type(dc_st)
        sd_round = zero_optax.opt_state_to_state_dict(dc_st)
        restored_dc = zero_optax.opt_state_from_state_dict(dc_st, sd_round)
        assert type(restored_dc) is type(dc_st)

    # Test ScaleByLBFGSState._make with non-empty memory
    lbfgs_mem_st = zero_optax.ScaleByLBFGSState._make(
        [
            1,
            params,
            updates,
            [params],
            [updates],
            [0.5, asarray(0.25)],
        ]
    )
    assert len(lbfgs_mem_st.weights_memory) == 2

    # Test GradientTransformationExtraArgs keyword construction (init=..., update=...)
    def _kw_init(params: Params) -> EmptyState:
        """Return EmptyState for keyword init test.

        Args:
            params: Parameters PyTree.

        Returns:
            EmptyState instance.
        """
        del params
        return EmptyState()

    def _kw_upd(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: zero_optax.base.ExtraArgValue,
    ) -> tuple[Updates, EmptyState]:
        """Return unchanged updates and EmptyState.

        Args:
            updates: Updates PyTree.
            state: Optimizer state.
            params: Optional parameters PyTree.
            **extra_args: Extra keyword arguments.

        Returns:
            Tuple `(updates, EmptyState())`.
        """
        del state, params, extra_args
        return updates, EmptyState()

    kw_extra_tx: GradientTransformationExtraArgs[EmptyState] = (
        GradientTransformationExtraArgs(init=_kw_init, update=_kw_upd)
    )
    u_kw, _ = kw_extra_tx.update(updates, kw_extra_tx.init(params), params)
    assert_allclose_optax(get_tree_item(u_kw, "w"), np.array([0.2, -0.4]))

    # Test adamw and lion with weight_decay=0.0 and params=None
    aw_zero = adamw(0.1, weight_decay=0.0)
    u_aw0, _ = aw_zero.update(updates, aw_zero.init(params), None)
    assert get_tree_item(u_aw0, "w").shape == (2,)
    lion_zero = lion(0.1, weight_decay=0.0)
    u_ln0, _ = lion_zero.update(updates, lion_zero.init(params), None)
    assert get_tree_item(u_ln0, "w").shape == (2,)

    # Test masked with mask_compatible_extra_args=True, tuple trees, and VariableLeaf masks
    m_compat = zero_optax.masked(
        kw_extra_tx,
        {"w": True, "m": False, "n": False},
        mask_compatible_extra_args=True,
    )
    u_mc, _ = m_compat.update(
        updates,
        m_compat.init(params),
        params,
        extra_tree=updates,
        non_matching=1.0,
    )
    assert_allclose_optax(get_tree_item(u_mc, "w"), np.array([0.2, -0.4]))

    from zero_optax.wrappers import (
        MaskedNode,
        _mask_tree_with_sentinel,
        _unmask_tree_with_fallback,
    )

    assert isinstance(
        _mask_tree_with_sentinel(asarray([1.0]), nnx.Param(False)), MaskedNode
    )
    assert isinstance(
        _mask_tree_with_sentinel((asarray([1.0]), asarray([2.0])), (True, False)),
        tuple,
    )
    assert _mask_tree_with_sentinel(asarray([1.0]), (True,)) is not None
    assert (
        _unmask_tree_with_fallback(asarray([1.0]), asarray([2.0]), nnx.Param(False))
        is not None
    )
    assert isinstance(
        _unmask_tree_with_fallback(
            (asarray([1.0]), MaskedNode()),
            (asarray([3.0]), asarray([4.0])),
            (True, False),
        ),
        tuple,
    )
    assert _unmask_tree_with_fallback(asarray([1.0]), asarray([2.0]), True) is not None
    assert isinstance(
        _unmask_tree_with_fallback(
            {"w": asarray([1.0]), "extra": asarray([2.0])},
            {"w": asarray([3.0]), "extra": asarray([4.0])},
            {"w": True},
        ),
        dict,
    )
    with pytest.raises(ValueError, match="Mask tree container structure"):
        _unmask_tree_with_fallback(asarray([1.0]), asarray([2.0]), (True,))

    # Test 1D per_example_global_norm_clip
    pe_1d, pe_cnt = zero_optax.per_example_global_norm_clip(
        {"v": asarray([3.0, 0.5])}, l2_norm_clip=1.0
    )
    assert int(pe_cnt.item()) == 1
    assert_allclose_optax(get_tree_item(pe_1d, "v"), np.array(1.5))

    # Test scale_by_zoom_linesearch with approx_dec_rtol=None and large stepsize_precision
    def _simple_q(p: Params) -> Tensor:
        """Simple 1D quadratic objective.

        Args:
            p: Parameters dict.

        Returns:
            Scalar objective Tensor.
        """
        if not isinstance(p, dict):
            return asarray(0.0)
        w_t = asarray(p["w"])
        return zero_optax._tree.reduce_sum(w_t * w_t)

    zm_no_approx = zero_optax.scale_by_zoom_linesearch(
        max_linesearch_steps=5,
        approx_dec_rtol=None,
        stepsize_precision=10.0,
        slope_rtol=0.5,
        curv_rtol=0.01,
    )
    p_q: dict[str, PyTree] = {"w": asarray([2.0, -2.0])}
    g_q: dict[str, PyTree] = {"w": asarray([-4.0, 4.0])}
    zm_no_approx.update(g_q, zm_no_approx.init(p_q), p_q, value_fn=_simple_q)

    # Test SM3 on 0D and tuple PyTrees
    sm3_tx = zero_optax.scale_by_sm3()
    p_tup_0d: PyTree = (asarray(2.0), [asarray([1.0, -2.0])], None)
    g_tup_0d: PyTree = (asarray(0.5), [asarray([0.2, -0.4])], None)
    st_sm3_0d = sm3_tx.init(p_tup_0d)
    u_sm3_0d, _ = sm3_tx.update(g_tup_0d, st_sm3_0d, p_tup_0d)
    assert isinstance(u_sm3_0d, tuple)


def test_phase5_end_to_end_flax_jax_chex_grain_orbax_and_lora_filter(
    tmp_path: Path,
) -> None:
    """End-to-end integration test of `zero_flax`, `zero_jax`, `zero_chex`, `zero_grain`, `zero_orbax`, and `LoRAParam` filters.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    from collections.abc import Hashable, Mapping

    import zero_grain
    import zero_jax
    from zero_flax.nnx.filterlib import Filter
    from zero_flax.nnx.variables import VariableBase
    from zero_grain._types import ElementTree
    from zero_jax._types import PyTree as JaxPyTree
    from zero_jax.numpy import ndarray as JaxArray
    from zero_orbax.checkpoint import StandardCheckpointer

    from zero_optax._types import StateDictTree

    # 1. End-to-end nnx.Linear + softmax_cross_entropy + value_and_grad + @jit + adamw + zero_chex + zero_orbax + zero_grain
    records: list[ElementTree] = [
        {
            "x": ensure_array([1.0, 0.5, -0.5, 0.2]),
            "y": ensure_array([1.0, 0.0]),
        },
        {
            "x": ensure_array([-0.5, 1.0, 0.3, -0.2]),
            "y": ensure_array([0.0, 1.0]),
        },
        {
            "x": ensure_array([0.8, -0.2, -0.4, 0.5]),
            "y": ensure_array([1.0, 0.0]),
        },
        {
            "x": ensure_array([-0.3, 0.9, 0.5, -0.4]),
            "y": ensure_array([0.0, 1.0]),
        },
    ]
    sampler = zero_grain.SequentialSampler(num_records=len(records))
    loader = zero_grain.DataLoader(
        data_source=records,
        sampler=sampler,
        operations=[zero_grain.BatchOperation(batch_size=2)],
    )
    grain_iter = iter(loader)

    sched = zero_optax.grain_warmup_cosine_decay_schedule(
        loader,
        init_value=1e-4,
        peak_value=1e-2,
        warmup_epochs=1.0,
        total_epochs=4.0,
    )
    tx = adamw(sched)

    linear_model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
    params_state = linear_model.state().split(nnx.Param)[0]
    opt_state = tx.init(params_state)

    def _linear_ce_loss(
        *args: JaxPyTree,
        **kwargs: JaxPyTree,
    ) -> JaxPyTree:
        """Compute cross-entropy loss for a linear parameter state.

        Args:
            *args: Positional arguments `(p_state, x_batch, y_batch)`.
            **kwargs: Unused keyword arguments.

        Returns:
            Scalar mean cross-entropy `Tensor`.
        """
        del kwargs
        p_state, x_batch, y_batch = args[0], args[1], args[2]
        assert isinstance(p_state, Mapping)
        w_leaf = p_state["kernel"]
        b_leaf = p_state["bias"]
        w_val = w_leaf.value if isinstance(w_leaf, nnx.Variable) else w_leaf
        b_val = b_leaf.value if isinstance(b_leaf, nnx.Variable) else b_leaf
        assert isinstance(x_batch, (JaxArray, Tensor))
        assert isinstance(y_batch, (JaxArray, Tensor))
        assert isinstance(w_val, (JaxArray, Tensor))
        assert isinstance(b_val, (JaxArray, Tensor))
        w_arr = ensure_array(w_val)
        b_arr = ensure_array(b_val)
        x_arr = ensure_array(x_batch)
        logits = zero_jax.numpy.matmul(x_arr, w_arr)
        assert isinstance(logits, JaxArray)
        logits_with_bias = asarray(logits) + asarray(b_arr)
        return zero_optax._tree.reduce_mean(
            zero_optax.softmax_cross_entropy(logits_with_bias, y_batch)
        )

    _traced_loss = zero_chex.assert_max_traces(_linear_ce_loss, n=2)
    jit_forward_loss = zero_jax.jit(_traced_loss)

    batch0 = next(grain_iter)
    assert isinstance(batch0, Mapping)
    x_raw = batch0["x"]
    y_raw = batch0["y"]
    assert isinstance(x_raw, (JaxArray, Tensor))
    assert isinstance(y_raw, (JaxArray, Tensor))
    x_b0 = ensure_array(x_raw)
    y_b0 = ensure_array(y_raw)

    w_init_leaf = params_state["kernel"]
    b_init_leaf = params_state["bias"]
    assert isinstance(w_init_leaf, nnx.Variable)
    assert isinstance(b_init_leaf, nnx.Variable)
    w_init_val = w_init_leaf.value
    b_init_val = b_init_leaf.value
    assert isinstance(w_init_val, (JaxArray, Tensor))
    assert isinstance(b_init_val, (JaxArray, Tensor))
    params_jax_tree: dict[str, JaxPyTree] = {
        "kernel": ensure_array(w_init_val),
        "bias": ensure_array(b_init_val),
    }

    jit_l0 = jit_forward_loss(params_jax_tree, x_b0, y_b0)
    loss_val, raw_grads = zero_jax.value_and_grad(_linear_ce_loss)(
        params_jax_tree, x_b0, y_b0
    )
    assert isinstance(jit_l0, (JaxArray, Tensor))
    assert isinstance(loss_val, (JaxArray, Tensor))
    assert isinstance(raw_grads, Mapping)
    gk = raw_grads["kernel"]
    gb = raw_grads["bias"]
    assert isinstance(gk, (JaxArray, Tensor))
    assert isinstance(gb, (JaxArray, Tensor))
    grads = nnx.State(
        {
            "kernel": nnx.Param(ensure_array(gk)),
            "bias": nnx.Param(ensure_array(gb)),
        }
    )
    zero_chex.assert_trees_all_finite(jit_l0)
    zero_chex.assert_trees_all_finite(loss_val)
    zero_chex.assert_trees_all_finite(grads)
    updates, opt_state = tx.update(grads, opt_state, params_state)
    updated_params = apply_updates(params_state, updates)
    assert isinstance(updated_params, nnx.State)
    params_state = updated_params
    zero_chex.assert_trees_all_finite(params_state)
    zero_chex.assert_trees_all_finite(opt_state)

    # Checkpoint (params, opt_state, grain_iter state) via zero_orbax StandardCheckpointer & PyTreeCheckpointer
    std_ckpt = StandardCheckpointer()
    std_dir = tmp_path / "std_ckpt"
    std_ckpt.save(std_dir, opt_state_to_state_dict(opt_state))
    std_restored_raw = std_ckpt.restore(std_dir)
    std_restored_opt = opt_state_from_state_dict(opt_state, std_restored_raw)
    zero_chex.assert_trees_all_close(opt_state, std_restored_opt)

    pytree_ckpt = PyTreeCheckpointer()
    bundle_dir = tmp_path / "bundle_ckpt"
    grain_state_dict = grain_iter.get_state()
    next_idx_raw = grain_state_dict.get("next_index", 0)
    grain_step = next_idx_raw if isinstance(next_idx_raw, int) else 0
    bundle_payload: StateDictTree = {
        "params": opt_state_to_state_dict(params_state),
        "opt_state": opt_state_to_state_dict(opt_state),
        "grain_step": grain_step,
    }
    pytree_ckpt.save(bundle_dir, bundle_payload)
    restored_bundle = pytree_ckpt.restore(bundle_dir)
    assert isinstance(restored_bundle, dict)
    restored_params = opt_state_from_state_dict(params_state, restored_bundle["params"])
    restored_opt = opt_state_from_state_dict(opt_state, restored_bundle["opt_state"])
    zero_chex.assert_trees_all_close(params_state, restored_params)
    zero_chex.assert_trees_all_close(opt_state, restored_opt)

    # 2. Test optax.masked and optax.partition / multi_transform with zero_flax.nnx.OfType(nnx.LoRAParam)
    lora_model = nnx.LoRALinear(4, 2, lora_rank=2, rngs=nnx.Rngs(0))
    lora_state = lora_model.state()
    lora_grads = tree_map(lambda x: ones_like(x) * 0.1, lora_state)

    lora_filter = nnx.OfType(nnx.LoRAParam)
    masked_lora_tx = zero_optax.masked(sgd(0.5), lora_filter)
    m_lora_st = masked_lora_tx.init(lora_state)
    u_lora, _ = masked_lora_tx.update(lora_grads, m_lora_st, lora_state)
    assert isinstance(u_lora, nnx.State)

    # Also test masked with callable returning Filter, and apply_mask_tree with Filter directly
    masked_callable_filter = zero_optax.masked(sgd(0.5), lambda _p: lora_filter)
    mc_st = masked_callable_filter.init(lora_state)
    masked_callable_filter.update(lora_grads, mc_st, lora_state)
    direct_filtered = apply_mask_tree(u_lora, lora_grads, lora_filter)
    assert isinstance(direct_filtered, nnx.State)
    assert _coerce_submask(lora_filter) is lora_filter
    assert _coerce_submask(nnx.LoRAParam) is nnx.LoRAParam

    # Test evaluate_filter_mask_tree on MappableDataclass, tuple, list, JaxArray, scalar, and Tensor
    from zero_optax._tree import evaluate_filter_mask_tree

    dc_mask = evaluate_filter_mask_tree(
        lora_filter, _ChexParamNode(w=asarray([1.0]), b=asarray([2.0]))
    )
    assert isinstance(dc_mask, tuple)
    tup_param: PyTree = (ensure_array([1.0]), [1.0, asarray(2.0)])
    tup_mask = evaluate_filter_mask_tree(lora_filter, tup_param)
    assert isinstance(tup_mask, tuple)

    # Test partition / multi_transform with Filter directly and with {label: Filter} mapping
    part_direct = zero_optax.multi_transform(
        {True: sgd(0.5), False: zero_optax.set_to_zero()},
        lora_filter,
    )
    pd_st = part_direct.init(lora_state)
    u_pd, _ = part_direct.update(lora_grads, pd_st, lora_state)
    assert isinstance(u_pd, nnx.State)

    part_callable_filter = zero_optax.partition(
        {True: sgd(0.5), False: zero_optax.set_to_zero()},
        lambda _p: lora_filter,
    )
    pcf_st = part_callable_filter.init(lora_state)
    part_callable_filter.update(lora_grads, pcf_st, lora_state)

    part_map_filters = zero_optax.multi_transform(
        {
            "lora": adamw(1e-2),
            "base": sgd(1e-3),
            "other": zero_optax.set_to_zero(),
        },
        {
            "lora": nnx.OfType(nnx.LoRAParam),
            "base": nnx.OfType(nnx.Param),
        },
    )
    pmf_st = part_map_filters.init(lora_state)
    u_pmf, _ = part_map_filters.update(lora_grads, pmf_st, lora_state)
    assert isinstance(u_pmf, nnx.State)

    part_map_all_filters = zero_optax.multi_transform(
        {
            "lora": adamw(1e-2),
            "base": sgd(1e-3),
        },
        {
            "lora": nnx.OfType(nnx.LoRAParam),
            "base": nnx.OfType(nnx.Param),
        },
    )
    pmaf_st = part_map_all_filters.init(lora_state)
    part_map_all_filters.update(lora_grads, pmaf_st, lora_state)

    # Test _resolve_filter_label_map_on_params and _build_label_mask on MappableDataclass, tuple, and list
    from zero_optax.combine import (
        _build_label_mask,
        _resolve_filter_label_map_on_params,
        _resolve_labels_for_params,
    )

    mixed_labels_map: dict[Hashable, Filter | type[VariableBase] | str] = {
        "lora": nnx.OfType(nnx.LoRAParam),
        "extra": "base",
    }
    assert (
        _resolve_labels_for_params(
            mixed_labels_map,
            lora_state,
            ["lora", "base"],
        )
        is not None
    )
    assert _resolve_labels_for_params(
        lambda _p: {"kernel": "base"},
        lora_state,
        ["base"],
    ) == {"kernel": "base"}

    dc_param = _ChexParamNode(w=asarray([1.0]), b=asarray([2.0]))
    dc_labels = _resolve_filter_label_map_on_params(
        {"lora": nnx.OfType(nnx.LoRAParam)}, "other", dc_param
    )
    assert _build_label_mask(dc_param, "other") is not None
    assert isinstance(dc_labels, dict)
    seq_labels = _resolve_filter_label_map_on_params(
        {"lora": nnx.OfType(nnx.LoRAParam)},
        None,
        ([asarray(1.0)], asarray(2.0)),
    )
    assert isinstance(seq_labels, tuple)
