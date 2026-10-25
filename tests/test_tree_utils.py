"""Comprehensive unit and parity tests for `zero_optax.tree_utils`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pytest
import zero_chex
from ml_switcheroo_compiler.core.dtype import DType
from zero_chex.dataclasses import MappableDataclassMixin
from zero_flax import nnx

import zero_optax
from tests.utils import assert_allclose_optax, get_tree_item
from zero_optax import tree_utils
from zero_optax._tree import asarray
from zero_optax._types import DTypeLike, Params, PyTree, Tensor
from zero_optax.projections._projections import _is_scalar_bound
from zero_optax.tree_utils._state_utils import KeyPath, NamedTupleKey


class _SampleTuple(NamedTuple):
    """Sample NamedTuple for PyTree structural and state path tests.

    Attributes:
        a: First tensor field.
        b: Second tensor field.
    """

    a: PyTree
    b: PyTree


@zero_chex.dataclass
class _SampleChexDC(MappableDataclassMixin):
    """Sample chex mappable dataclass for PyTree tests.

    Attributes:
        x: Tensor field.
    """

    x: Tensor


def test_tree_math_arithmetic_and_clipping() -> None:
    """Test arithmetic, scaling, clipping, conj, real, and where operations."""
    t1: dict[str, PyTree] = {"w": asarray([1.0, -2.0]), "n": None}
    t2: dict[str, PyTree] = {"w": asarray([3.0, 4.0]), "n": None}
    t3: dict[str, PyTree] = {"w": asarray([0.5, 0.5]), "n": None}

    added = tree_utils.tree_add(t1, t2, t3)
    assert_allclose_optax(get_tree_item(added, "w"), np.array([4.5, 2.5]))

    subbed = tree_utils.tree_sub(t1, t2)
    assert_allclose_optax(get_tree_item(subbed, "w"), np.array([-2.0, -6.0]))

    mulled = tree_utils.tree_mul(t1, t2)
    assert_allclose_optax(get_tree_item(mulled, "w"), np.array([3.0, -8.0]))

    divved = tree_utils.tree_div(t1, t2)
    assert_allclose_optax(get_tree_item(divved, "w"), np.array([1.0 / 3.0, -0.5]))

    scaled = tree_utils.tree_scalar_mul(2.0, t1)
    assert_allclose_optax(get_tree_item(scaled, "w"), np.array([2.0, -4.0]))

    scaled2 = tree_utils.tree_scale(3.0, t1)
    assert_allclose_optax(get_tree_item(scaled2, "w"), np.array([3.0, -6.0]))

    add_scaled = tree_utils.tree_add_scale(t1, 0.5, t2)
    assert_allclose_optax(get_tree_item(add_scaled, "w"), np.array([2.5, 0.0]))

    clipped_both = tree_utils.tree_clip(t1, min_value=-1.0, max_value=0.5)
    assert_allclose_optax(get_tree_item(clipped_both, "w"), np.array([0.5, -1.0]))
    clipped_none = tree_utils.tree_clip(t1, min_value=None, max_value=None)
    assert_allclose_optax(get_tree_item(clipped_none, "w"), np.array([1.0, -2.0]))

    conj_t = tree_utils.tree_conj(t1)
    assert_allclose_optax(get_tree_item(conj_t, "w"), np.array([1.0, -2.0]))

    real_t = tree_utils.tree_real(t1)
    assert_allclose_optax(get_tree_item(real_t, "w"), np.array([1.0, -2.0]))

    w_true = tree_utils.tree_where(True, t1, t2)
    assert_allclose_optax(get_tree_item(w_true, "w"), np.array([1.0, -2.0]))
    w_false = tree_utils.tree_where(False, t1, t2)
    assert_allclose_optax(get_tree_item(w_false, "w"), np.array([3.0, 4.0]))


def test_tree_creation_casting_dtype_and_batch_shape() -> None:
    """Test zeros_like, ones_like, full_like, cast, cast_like, dtype, and batch_shape."""
    t1: dict[str, PyTree] = {"w": asarray([1.0, 2.0]), "n": None}

    z_tree = tree_utils.tree_zeros_like(t1)
    assert_allclose_optax(get_tree_item(z_tree, "w"), np.array([0.0, 0.0]))

    o_tree = tree_utils.tree_ones_like(t1)
    assert_allclose_optax(get_tree_item(o_tree, "w"), np.array([1.0, 1.0]))

    f_tree = tree_utils.tree_full_like(t1, 7.0)
    assert_allclose_optax(get_tree_item(f_tree, "w"), np.array([7.0, 7.0]))

    assert tree_utils.tree_cast(t1, None) is t1
    cast_i32 = tree_utils.tree_cast(t1, "int32")
    assert get_tree_item(cast_i32, "w").dtype == DType.int32

    cast_like_res = tree_utils.tree_cast_like(t1, cast_i32)
    assert get_tree_item(cast_like_res, "w").dtype == DType.int32

    assert tree_utils.tree_dtype({"n": None}) == DType.float32
    assert tree_utils.tree_dtype(cast_i32) == DType.int32
    mixed: dict[str, PyTree] = {
        "a": asarray([1.0], dtype="float32"),
        "b": asarray([1], dtype="int32"),
        "c": asarray([2.0], dtype="float32"),
    }
    with pytest.raises(ValueError, match="Multiple dtypes"):
        tree_utils.tree_dtype(mixed, None)
    assert tree_utils.tree_dtype(mixed, "highest") == DType.float32
    assert tree_utils.tree_dtype(mixed, "promote") == DType.float32
    assert tree_utils.tree_dtype(mixed, "lowest") == DType.int32

    def _custom_dt(dts: Sequence[DType]) -> DType:
        """Select first dtype from sequence.

        Args:
            dts: Sequence of unique dtypes.

        Returns:
            First `DType`.
        """
        return dts[0]

    assert tree_utils.tree_dtype(mixed, _custom_dt) == DType.float32
    with pytest.raises(ValueError, match="Unsupported mixed_dtype_handler"):
        tree_utils.tree_dtype(mixed, "invalid_mode")

    assert tree_utils.tree_batch_shape(t1, ()) is t1
    batched = tree_utils.tree_batch_shape(t1, (2, 3))
    assert get_tree_item(batched, "w").shape == (2, 3, 2)
    empty_t = asarray([])
    batched_empty = tree_utils.tree_batch_shape({"e": empty_t}, (2,))
    assert get_tree_item(batched_empty, "e").shape == (2, 0)


def test_tree_reductions_norms_and_allclose() -> None:
    """Test sum, max, min, size, vdot, norms, and allclose."""
    t1: dict[str, PyTree] = {"a": asarray([1.0, -3.0]), "b": asarray([4.0]), "n": None}
    t2: dict[str, PyTree] = {"a": asarray([2.0, 1.0]), "b": asarray([-1.0]), "n": None}

    assert float(tree_utils.tree_sum(t1).item()) == pytest.approx(2.0)
    assert float(tree_utils.tree_max(t1).item()) == pytest.approx(4.0)
    assert float(tree_utils.tree_min(t1).item()) == pytest.approx(-3.0)
    assert math.isinf(float(tree_utils.tree_max({"n": None}).item()))
    assert math.isinf(float(tree_utils.tree_min({"n": None}).item()))
    assert tree_utils.tree_size(t1) == 3
    assert float(tree_utils.tree_vdot(t1, t2).item()) == pytest.approx(-5.0)

    assert float(tree_utils.tree_l1_norm(t1).item()) == pytest.approx(8.0)
    assert float(tree_utils.tree_l2_norm(t1, squared=True).item()) == pytest.approx(
        26.0
    )
    assert float(tree_utils.tree_l2_norm(t1, squared=False).item()) == pytest.approx(
        math.sqrt(26.0)
    )
    assert float(tree_utils.tree_linf_norm(t1).item()) == pytest.approx(4.0)
    assert float(tree_utils.tree_linf_norm({"n": None}).item()) == pytest.approx(0.0)

    assert float(tree_utils.tree_norm(t1, ord=None).item()) == pytest.approx(
        math.sqrt(26.0)
    )
    assert float(
        tree_utils.tree_norm(t1, ord=1, squared=False).item()
    ) == pytest.approx(8.0)
    assert float(tree_utils.tree_norm(t1, ord=1, squared=True).item()) == pytest.approx(
        64.0
    )
    assert float(
        tree_utils.tree_norm(t1, ord="inf", squared=False).item()
    ) == pytest.approx(4.0)
    assert float(
        tree_utils.tree_norm(t1, ord=float("inf"), squared=True).item()
    ) == pytest.approx(16.0)
    with pytest.raises(ValueError, match="Unsupported norm ord"):
        tree_utils.tree_norm(t1, ord=3)

    assert tree_utils.tree_allclose(t1, t1)
    assert not tree_utils.tree_allclose(t1, t2)
    assert not tree_utils.tree_allclose(t1, [asarray([1.0])])
    assert not tree_utils.tree_allclose(
        {"a": asarray([1.0, 2.0])}, {"a": asarray([1.0])}
    )
    assert not tree_utils.tree_allclose({"a": None}, {"a": asarray([1.0])})
    nan_tree = {"a": asarray([float("nan")])}
    assert not tree_utils.tree_allclose(nan_tree, nan_tree, equal_nan=False)
    assert tree_utils.tree_allclose(nan_tree, nan_tree, equal_nan=True)
    assert not tree_utils.tree_allclose(nan_tree, {"a": asarray([1.0])}, equal_nan=True)
    assert not tree_utils.tree_allclose(
        {"a": asarray([float("nan"), 1.0])},
        {"a": asarray([float("nan"), 2.0])},
        equal_nan=True,
    )

    st_adam = zero_optax.adam(0.1).init({"w": asarray([1.0])})
    assert tree_utils.tree_allclose(st_adam, st_adam)
    dc_obj = _SampleChexDC(x=asarray([1.0]))
    assert tree_utils.tree_allclose(dc_obj, dc_obj)
    nt_obj = _SampleTuple(a=asarray([1.0]), b=asarray([2.0]))
    assert tree_utils.tree_allclose(nt_obj, nt_obj)
    assert tree_utils.tree_allclose([asarray([1.0])], [asarray([1.0])])


def test_tree_moments_and_random_utils() -> None:
    """Test moment updates, bias correction, and random key utilities."""
    u: dict[str, PyTree] = {"w": asarray([2.0, -3.0]), "n": None}
    m: dict[str, PyTree] = {"w": asarray([1.0, 1.0]), "n": None}

    m1 = tree_utils.tree_update_moment(u, m, decay=0.9, order=1)
    assert_allclose_optax(get_tree_item(m1, "w"), np.array([1.1, 0.6]))

    m2 = tree_utils.tree_update_moment_per_elem_norm(u, m, decay=0.9, order=2)
    assert_allclose_optax(get_tree_item(m2, "w"), np.array([1.3, 1.8]))

    minf = tree_utils.tree_update_infinity_moment(u, m, decay=0.5, eps=0.0)
    assert_allclose_optax(get_tree_item(minf, "w"), np.array([2.0, 3.0]))

    bc = tree_utils.tree_bias_correction(m, decay=0.5, count=asarray(1))
    assert_allclose_optax(get_tree_item(bc, "w"), np.array([2.0, 2.0]))
    bc_zero = tree_utils.tree_bias_correction(m, decay=1.0, count=0)
    assert_allclose_optax(get_tree_item(bc_zero, "w"), np.array([1.0, 1.0]))

    complex_tree: dict[str, PyTree] = {
        "d": {"x": asarray([1.0, 2.0]), "s": asarray(3.0), "e": asarray([])},
        "l": [asarray([4.0])],
        "t": (asarray([5.0]),),
        "dc": _SampleChexDC(x=asarray([6.0])),
        "n": None,
    }
    keys1 = tree_utils.tree_split_key_like(42, complex_tree)
    keys2 = tree_utils.tree_split_key_like(asarray([0, 42]), complex_tree)
    assert isinstance(keys1, dict) and isinstance(keys2, dict)

    rand_tree = tree_utils.tree_random_like(42, complex_tree)
    assert isinstance(rand_tree, dict)

    def _const_sampler(k: Tensor, shape: tuple[int, ...], dt: DTypeLike) -> Tensor:
        """Return ones of `shape` and `dt`.

        Args:
            k: Key tensor.
            shape: Target shape.
            dt: Target dtype.

        Returns:
            Ones tensor.
        """
        del k
        return zero_optax._tree.ones(shape, dtype=dt)

    rand_custom = tree_utils.tree_random_like(
        42, {"w": asarray([1.0, 2.0])}, sampler=_const_sampler, dtype="float32"
    )
    assert_allclose_optax(get_tree_item(rand_custom, "w"), np.array([1.0, 1.0]))

    unwrapped = tree_utils.tree_unwrap_random_key_data(
        {
            "v": nnx.Param(1.0),
            "t": asarray([2.0]),
            "n": None,
        }
    )
    assert_allclose_optax(get_tree_item(unwrapped, "t"), np.array([2.0]))


def test_tree_state_utils_get_set_and_map_params() -> None:
    """Test NamedTupleKey, tree_get_all_with_path, tree_get, tree_set, and tree_map_params."""
    nt_key = NamedTupleKey("ScaleByAdamState", "count")
    assert str(nt_key) == "ScaleByAdamState.count"

    params: dict[str, PyTree] = {"w": asarray([1.0, 2.0])}
    opt = zero_optax.chain(zero_optax.adam(0.1), zero_optax.sgd(0.01, momentum=0.9))
    state = opt.init(params)

    all_counts = tree_utils.tree_get_all_with_path(state, "count")
    assert len(all_counts) >= 2
    with pytest.raises(KeyError, match="Found multiple values"):
        tree_utils.tree_get(state, "count")

    def _filter_adam(path: KeyPath, val: PyTree) -> bool:
        """Select only `ScaleByAdamState.count`.

        Args:
            path: KeyPath to value.
            val: Candidate value.

        Returns:
            True if parent is `ScaleByAdamState`.
        """
        del val
        return any(
            isinstance(entry, NamedTupleKey) and entry.tuple_name == "ScaleByAdamState"
            for entry in path
        )

    adam_count = tree_utils.tree_get(state, "count", filtering=_filter_adam)
    assert isinstance(adam_count, Tensor) and int(adam_count.item()) == 0
    assert tree_utils.tree_get(state, "nonexistent", default=99) == 99

    adam_node = tree_utils.tree_get(state, "ScaleByAdamState")
    assert adam_node is not None

    updated_state = tree_utils.tree_set(state, _filter_adam, count=asarray(5))
    new_cnt = tree_utils.tree_get(updated_state, "count", filtering=_filter_adam)
    assert isinstance(new_cnt, Tensor) and int(new_cnt.item()) == 5

    replaced_adam = tree_utils.tree_set(state, ScaleByAdamState=adam_node)
    assert replaced_adam is not None

    list_dict_tree: list[PyTree] = [{"lr": asarray(0.1)}, {"other": asarray(0.2)}]
    updated_ld = tree_utils.tree_set(list_dict_tree, lr=asarray(0.5))
    assert isinstance(updated_ld, list)

    with pytest.raises(KeyError, match="Found no values matching"):
        tree_utils.tree_set(state, missing_key=1)
    with pytest.raises(KeyError, match="given the filtering operation"):
        tree_utils.tree_set(state, lambda p, v: False, count=1)

    def _double_param(p: PyTree) -> PyTree:
        """Double parameter tensor leaf.

        Args:
            p: Parameter tensor leaf.

        Returns:
            Doubled tensor.
        """
        return asarray(p) * 2.0 if _is_scalar_bound(p) else p

    def _add_two(p: PyTree, a: PyTree) -> PyTree:
        """Add two parameter leaves.

        Args:
            p: Primary leaf.
            a: Extra leaf 1.

        Returns:
            Sum tensor.
        """
        return (
            asarray(p) + asarray(a)
            if _is_scalar_bound(p) and _is_scalar_bound(a)
            else p
        )

    def _add_three(p: PyTree, a: PyTree, b: PyTree) -> PyTree:
        """Add three parameter leaves.

        Args:
            p: Primary leaf.
            a: Extra leaf 1.
            b: Extra leaf 2.

        Returns:
            Sum tensor.
        """
        return (
            asarray(p) + asarray(a) + asarray(b)
            if _is_scalar_bound(p) and _is_scalar_bound(a) and _is_scalar_bound(b)
            else p
        )

    def _add_four(p: PyTree, a: PyTree, b: PyTree, c: PyTree) -> PyTree:
        """Add four parameter leaves.

        Args:
            p: Primary leaf.
            a: Extra leaf 1.
            b: Extra leaf 2.
            c: Extra leaf 3.

        Returns:
            Sum tensor.
        """
        return (
            asarray(p) + asarray(a) + asarray(b) + asarray(c)
            if _is_scalar_bound(p)
            and _is_scalar_bound(a)
            and _is_scalar_bound(b)
            and _is_scalar_bound(c)
            else p
        )

    mapped_1 = zero_optax.tree_map_params(opt, _double_param, state)
    assert mapped_1 is not None
    mapped_2 = zero_optax.tree_map_params(
        opt.init,
        _add_two,
        state,
        params,
        transform_non_params=lambda x: x,
        is_leaf=lambda x: isinstance(x, Tensor),
    )
    assert mapped_2 is not None
    mapped_3 = zero_optax.tree_map_params(opt, _add_three, state, params, params)
    assert mapped_3 is not None
    mapped_4 = zero_optax.tree_map_params(opt, _add_four, state, params, params, params)
    assert mapped_4 is not None
    with pytest.raises(ValueError, match="supports up to 3 extra trees"):
        zero_optax.tree_map_params(
            opt, _add_four, state, params, params, params, params
        )

    def _custom_init(p: Params) -> PyTree:
        """Custom init returning dict, list, and tuple containers with extra non-param key.

        Args:
            p: Parameters PyTree.

        Returns:
            Composite state dictionary.
        """
        return {"p": p, "lst": [p, 1], "tup": (p,)}

    custom_state: dict[str, PyTree] = {
        "p": {
            "dc": _SampleChexDC(x=asarray([1.0])),
            "l": [asarray([2.0])],
            "t": (asarray([3.0]),),
        },
        "lst": [
            {
                "dc": _SampleChexDC(x=asarray([1.0])),
                "l": [asarray([2.0])],
                "t": (asarray([3.0]),),
            },
            1,
        ],
        "tup": (
            {
                "dc": _SampleChexDC(x=asarray([1.0])),
                "l": [asarray([2.0])],
                "t": (asarray([3.0]),),
            },
        ),
        "extra_meta": 42,
    }
    mapped_custom = zero_optax.tree_map_params(
        _custom_init,
        _double_param,
        custom_state,
        transform_non_params=lambda x: x,
    )
    assert isinstance(mapped_custom, dict)
    mapped_custom_no_tnp = zero_optax.tree_map_params(
        _custom_init,
        _double_param,
        custom_state,
    )
    assert isinstance(mapped_custom_no_tnp, dict)
