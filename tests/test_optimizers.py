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
    _to_chex_tree,
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
    assert apply_mask_tree(1.0, 2.0, {"unused": True}) == 1.0

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
        **extra_args: float | int | str | bool | Tensor | None,
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
        zero_chex.assert_trees_all_equal_structs(
            _to_chex_tree(next_opt_state), _to_chex_tree(restored_st)
        )
        zero_chex.assert_trees_all_close(
            _to_chex_tree(next_opt_state), _to_chex_tree(restored_st)
        )

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
    fdict = FrozenDict({"w": asarray([1.0, 2.0]), "b": asarray([0.5])})
    mapped_fdict = tree_map(lambda x: asarray(x) * 2.0, fdict)
    assert isinstance(mapped_fdict, FrozenDict)
    assert len(tree_leaves(fdict)) == 2
    masked_fdict = apply_mask_tree(
        mapped_fdict,
        fdict,
        FrozenDict({"w": True, "b": (False, True)}),
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
    assert _to_chex_tree(dc_sum) is not None
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
    zero_chex.assert_trees_all_equal_structs(
        _to_chex_tree(c_state1), _to_chex_tree(c_state1_restored)
    )
    zero_chex.assert_trees_all_close(
        _to_chex_tree(c_state1), _to_chex_tree(c_state1_restored)
    )

    u_expected, c_state2_expected = chained_tx.update(
        grads_state, c_state1, params_state
    )
    u_restored, c_state2_restored = chained_tx.update(
        grads_state, c_state1_restored, params_state
    )
    zero_chex.assert_trees_all_close(
        _to_chex_tree(u_expected), _to_chex_tree(u_restored)
    )
    zero_chex.assert_trees_all_close(
        _to_chex_tree(c_state2_expected), _to_chex_tree(c_state2_restored)
    )

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
    assert _to_chex_tree(list_state) is not None
    assert isinstance(
        opt_state_from_state_dict(sched_st, (asarray(2),)), ScaleByScheduleState
    )
    assert isinstance(opt_state_from_state_dict(fdict, fdict), FrozenDict)
    assert isinstance(opt_state_from_state_dict(param_leaf, param_leaf), nnx.Param)
    assert opt_state_to_state_dict(z_arr) is not None
    assert _to_chex_tree(z_arr) is not None
    assert _to_raw_array_input(None) == 0.0
    assert _coerce_pytree_child([True, False]) == [True, False]
    assert _coerce_pytree_child("not_a_pytree") is None
    assert _unwrap_orbax_state_dict([1, 2]) == [1, 2]
    assert _unwrap_orbax_state_dict("not_a_pytree") is None
    assert opt_state_to_state_dict(42) == 42
    assert opt_state_from_state_dict(42, 99) == 99
    with pytest.raises(TypeError, match="Cannot restore"):
        opt_state_from_state_dict(sched_st, 123)
