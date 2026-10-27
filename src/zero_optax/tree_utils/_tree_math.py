"""PyTree math, moment updates, casting, and norm utilities for `zero_optax.tree_utils`."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from zero_optax._tree import (
    _coerce_pytree_child,
    abs_tensor,
    asarray,
    expand_dims,
    full_like,
    maximum,
    minimum,
    ones,
    power,
    reduce_max,
    reduce_min,
    reduce_sum,
    reshape,
    tree_leaves,
    tree_map,
    where,
)
from zero_optax._tree import (
    tree_add as _base_tree_add,
)
from zero_optax._tree import (
    tree_l2_norm as _base_tree_l2_norm,
)
from zero_optax._tree import (
    tree_mul as _base_tree_mul,
)
from zero_optax._tree import (
    tree_ones_like as _base_tree_ones_like,
)
from zero_optax._tree import (
    tree_scale as _base_tree_scale,
)
from zero_optax._tree import (
    tree_sub as _base_tree_sub,
)
from zero_optax._tree import (
    tree_vdot as _base_tree_vdot,
)
from zero_optax._tree import (
    tree_zeros_like as _base_tree_zeros_like,
)
from zero_optax._tree import (
    where as _where_tensor,
)
from zero_optax._types import (
    ArrayLike,
    CustomMappingPyTreeProtocol,
    DType,
    DTypeLike,
    NamedTupleProtocol,
    OptStateProtocol,
    PyTree,
    Scalar,
    Tensor,
    TensorLeaf,
    Updates,
)
from zero_optax.base import OptState

_DTYPE_PROMOTION_ORDER: dict[str, int] = {
    "bool": 0,
    "uint8": 1,
    "int8": 2,
    "uint16": 3,
    "int16": 4,
    "uint32": 5,
    "int32": 6,
    "uint64": 7,
    "int64": 8,
    "bfloat16": 9,
    "float16": 10,
    "float32": 11,
    "float64": 12,
    "complex64": 13,
    "complex128": 14,
}


def _flatten_tensor_floats(t: Tensor) -> list[float]:
    """Flatten a `Tensor` into a 1D Python list of floats.

    Args:
        t: Input `Tensor`.

    Returns:
        Flat list of floats.
    """
    prod = 1
    for dim in t.shape:
        prod *= dim
    flat = reshape(t, (prod,)) if prod > 0 else t
    return [float(flat[i].item()) for i in range(prod)]


def tree_add(
    tree_x: PyTree,
    tree_y: PyTree,
    *other_trees: PyTree,
) -> PyTree:
    """Add two or more PyTrees element-wise.

    Args:
        tree_x: First PyTree.
        tree_y: Second PyTree.
        *other_trees: Optional additional PyTrees to add.

    Returns:
        PyTree containing the element-wise sum of all input PyTrees.
    """
    return _base_tree_add(tree_x, tree_y, *other_trees)


def tree_sub(tree_x: PyTree, tree_y: PyTree) -> PyTree:
    """Subtract `tree_y` from `tree_x` element-wise.

    Args:
        tree_x: First PyTree (minuend).
        tree_y: Second PyTree (subtrahend).

    Returns:
        PyTree containing `tree_x - tree_y`.
    """
    return _base_tree_sub(tree_x, tree_y)


def tree_mul(tree_x: PyTree, tree_y: PyTree) -> PyTree:
    """Multiply two PyTrees element-wise.

    Args:
        tree_x: First PyTree.
        tree_y: Second PyTree.

    Returns:
        PyTree containing the element-wise product `tree_x * tree_y`.
    """
    return _base_tree_mul(tree_x, tree_y)


def tree_div(tree_x: PyTree, tree_y: PyTree) -> PyTree:
    """Divide `tree_x` by `tree_y` element-wise.

    Args:
        tree_x: Numerator PyTree.
        tree_y: Denominator PyTree.

    Returns:
        PyTree containing the element-wise quotient `tree_x / tree_y`.
    """

    def _div_pair(a: TensorLeaf, b: TensorLeaf) -> TensorLeaf:
        """Divide `a` by `b`.

        Args:
            a: Numerator leaf.
            b: Denominator leaf.

        Returns:
            Quotient leaf `a / b` (or None if `a` is None).
        """
        if a is None or b is None:
            return None
        return asarray(a) / asarray(b)

    return tree_map(_div_pair, tree_x, tree_y)


def tree_scalar_mul(scalar: Scalar, tree: PyTree) -> PyTree:
    """Multiply every leaf in `tree` by `scalar`.

    Args:
        scalar: Scalar multiplier.
        tree: Input PyTree.

    Returns:
        Scaled PyTree `scalar * tree`.
    """
    return _base_tree_scale(scalar, tree)


def tree_scale(scalar: Scalar, tree: PyTree) -> PyTree:
    """Multiply a PyTree by a scalar (`out = scalar * tree`).

    Args:
        scalar: Scalar multiplier.
        tree: Input PyTree.

    Returns:
        Scaled PyTree `scalar * tree`.
    """
    return tree_scalar_mul(scalar, tree)


def tree_add_scale(
    tree_x: PyTree,
    scalar: Scalar,
    tree_y: PyTree,
) -> PyTree:
    """Compute `tree_x + scalar * tree_y` leaf-wise.

    Args:
        tree_x: First PyTree.
        scalar: Scalar multiplier for `tree_y`.
        tree_y: Second PyTree to scale and add.

    Returns:
        PyTree with `tree_x + scalar * tree_y`.
    """
    s_t = asarray(scalar)

    def _add_scaled(x: TensorLeaf, y: TensorLeaf) -> TensorLeaf:
        """Compute `x + scalar * y` for a leaf pair.

        Args:
            x: First leaf.
            y: Second leaf.

        Returns:
            Combined leaf (or None if `x` or `y` is None).
        """
        if x is None or y is None:
            return None
        return asarray(x) + s_t * asarray(y)

    return tree_map(_add_scaled, tree_x, tree_y)


def tree_clip(
    tree: PyTree,
    min_value: Scalar | None = None,
    max_value: Scalar | None = None,
) -> PyTree:
    """Clip all leaves in `tree` to `[min_value, max_value]`.

    Args:
        tree: Input PyTree.
        min_value: Optional minimum lower bound.
        max_value: Optional maximum upper bound.

    Returns:
        Clipped PyTree with identical structure.
    """

    def _clip_leaf(x: TensorLeaf) -> TensorLeaf:
        """Clip leaf `x` to `[min_value, max_value]`.

        Args:
            x: Input leaf.

        Returns:
            Clipped Tensor leaf.
        """
        if x is None:
            return None
        res = asarray(x)
        if min_value is not None:
            res = maximum(res, min_value)
        if max_value is not None:
            res = minimum(res, max_value)
        return res

    return tree_map(_clip_leaf, tree)


def tree_conj(tree: PyTree) -> PyTree:
    """Compute the complex conjugate of every leaf in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Conjugated PyTree with identical structure (returns real tensors unchanged).
    """

    def _conj_leaf(x: TensorLeaf) -> TensorLeaf:
        """Return conjugate of leaf `x` (real-valued tensor leaves are returned as-is).

        Args:
            x: Input leaf.

        Returns:
            Conjugated Tensor leaf.
        """
        if x is None:
            return None
        t = asarray(x)
        return asarray(t, dtype=t.dtype)

    return tree_map(_conj_leaf, tree)


def tree_real(tree: PyTree) -> PyTree:
    """Compute the real part of every leaf in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Real-valued PyTree with identical structure.
    """

    def _real_leaf(x: TensorLeaf) -> TensorLeaf:
        """Return real part of leaf `x`.

        Args:
            x: Input leaf.

        Returns:
            Real Tensor leaf.
        """
        if x is None:
            return None
        t = asarray(x)
        return asarray(t, dtype=t.dtype)

    return tree_map(_real_leaf, tree)


def tree_where(
    condition: ArrayLike,
    tree_x: PyTree,
    tree_y: PyTree,
) -> PyTree:
    """Select `tree_x` values where `condition` is True else `tree_y` values.

    Args:
        condition: Boolean scalar or tensor mask.
        tree_x: PyTree selected where `condition` is True.
        tree_y: PyTree selected where `condition` is False.

    Returns:
        Selected PyTree matching the structure of `tree_x` and `tree_y`.
    """
    cond_t = asarray(condition)

    def _select_leaf(x: TensorLeaf, y: TensorLeaf) -> TensorLeaf:
        """Select between leaf `x` and leaf `y` using `cond_t`.

        Args:
            x: True-branch leaf.
            y: False-branch leaf.

        Returns:
            Selected Tensor leaf.
        """
        if x is None or y is None:
            return x if bool(cond_t.item()) else y
        return _where_tensor(cond_t, x, y)

    return tree_map(_select_leaf, tree_x, tree_y)


def tree_zeros_like(
    tree: PyTree,
    dtype: DTypeLike = None,
) -> PyTree:
    """Create an all-zeros PyTree with the same structure as `tree`.

    Args:
        tree: Template PyTree.
        dtype: Optional data type override for the zero leaves.

    Returns:
        All-zeros PyTree.
    """
    return _base_tree_zeros_like(tree, dtype=dtype)


def tree_ones_like(
    tree: PyTree,
    dtype: DTypeLike = None,
) -> PyTree:
    """Create an all-ones PyTree with the same structure as `tree`.

    Args:
        tree: Template PyTree.
        dtype: Optional data type override for the ones leaves.

    Returns:
        All-ones PyTree.
    """
    return _base_tree_ones_like(tree, dtype=dtype)


def tree_full_like(
    tree: PyTree,
    fill_value: Scalar,
    dtype: DTypeLike = None,
) -> PyTree:
    """Create a PyTree with the same structure as `tree` filled with `fill_value`.

    Args:
        tree: Template PyTree.
        fill_value: Scalar fill value for all leaves.
        dtype: Optional data type override.

    Returns:
        Filled PyTree.
    """

    def _fill_leaf(x: TensorLeaf) -> TensorLeaf:
        """Create a constant-filled tensor matching `x`.

        Args:
            x: Input leaf.

        Returns:
            Filled Tensor leaf (or None if `x` is None).
        """
        return None if x is None else full_like(x, fill_value, dtype=dtype)

    return tree_map(_fill_leaf, tree)


def tree_cast(
    tree: PyTree,
    dtype: DTypeLike = None,
) -> PyTree:
    """Cast all leaves in `tree` to `dtype` (or return unchanged if `dtype is None`).

    Args:
        tree: Input PyTree to cast.
        dtype: Target data type, or None to skip casting.

    Returns:
        PyTree with leaves cast to `dtype`.
    """
    if dtype is None:
        return tree

    def _cast_leaf(x: TensorLeaf) -> TensorLeaf:
        """Cast leaf `x` to `dtype`.

        Args:
            x: Input leaf.

        Returns:
            Cast Tensor leaf.
        """
        return None if x is None else asarray(x, dtype=dtype)

    return tree_map(_cast_leaf, tree)


def tree_cast_like(
    tree: PyTree,
    other_tree: PyTree,
) -> PyTree:
    """Cast each leaf of `tree` to the dtype of the corresponding leaf in `other_tree`.

    Args:
        tree: Input PyTree to cast.
        other_tree: Reference PyTree providing target leaf dtypes.

    Returns:
        PyTree with leaves cast to match `other_tree` dtypes.
    """

    def _cast_pair(x: TensorLeaf, ref: TensorLeaf) -> TensorLeaf:
        """Cast leaf `x` to the dtype of `ref`.

        Args:
            x: Input leaf.
            ref: Reference leaf.

        Returns:
            Cast Tensor leaf.
        """
        if x is None or ref is None:
            return x
        ref_t = asarray(ref)
        return asarray(x, dtype=ref_t.dtype)

    return tree_map(_cast_pair, tree, other_tree)


def tree_dtype(
    tree: PyTree,
    mixed_dtype_handler: str | Callable[[Sequence[DType]], DType] | None = None,
) -> DType:
    """Fetch or resolve the dtype of `tree`.

    Args:
        tree: Input PyTree to inspect.
        mixed_dtype_handler: Strategy for handling multiple dtypes across leaves:
            `None` (require identical dtypes), `'promote'`, `'highest'`, `'lowest'`,
            or a custom callable.

    Returns:
        Resolved `DType` of the PyTree.

    Raises:
        ValueError: If `mixed_dtype_handler` is `None` and multiple dtypes are present,
            or if an unsupported `mixed_dtype_handler` is supplied.
    """
    leaves = [asarray(leaf) for leaf in tree_leaves(tree) if leaf is not None]
    if not leaves:
        return DType.float32
    dtypes = [leaf.dtype for leaf in leaves]
    unique_dtypes: list[DType] = []
    for dt in dtypes:
        if dt not in unique_dtypes:
            unique_dtypes.append(dt)
    if len(unique_dtypes) == 1:
        return unique_dtypes[0]
    if callable(mixed_dtype_handler):
        return mixed_dtype_handler(unique_dtypes)
    if mixed_dtype_handler is None:
        raise ValueError(
            f"Multiple dtypes found in tree: {unique_dtypes}. Specify mixed_dtype_handler."
        )
    if mixed_dtype_handler in ("highest", "promote"):
        return max(
            unique_dtypes,
            key=lambda d: _DTYPE_PROMOTION_ORDER.get(d.value, 11),
        )
    if mixed_dtype_handler == "lowest":
        return min(
            unique_dtypes,
            key=lambda d: _DTYPE_PROMOTION_ORDER.get(d.value, 11),
        )
    raise ValueError(f"Unsupported mixed_dtype_handler: {mixed_dtype_handler!r}.")


def tree_batch_shape(
    tree: PyTree,
    shape: tuple[int, ...] = (),
) -> PyTree:
    """Prepend leading batch dimensions `shape` to every leaf of `tree`.

    Args:
        tree: Input PyTree.
        shape: Leading batch dimensions to broadcast/prepend to each leaf.

    Returns:
        PyTree with each leaf broadcast to `(*shape, *leaf.shape)`.
    """
    if not shape:
        return tree

    def _batch_leaf(x: TensorLeaf) -> TensorLeaf:
        """Broadcast leaf `x` to include leading dimensions `shape`.

        Args:
            x: Input leaf.

        Returns:
            Broadcast Tensor leaf.
        """
        if x is None:
            return None
        t = asarray(x)
        expanded = t
        for _ in shape:
            expanded = expand_dims(expanded, 0)
        prefix_ones = ones((*shape, *(1 for _ in t.shape)), dtype=t.dtype)
        return asarray(prefix_ones * expanded, dtype=t.dtype)

    return tree_map(_batch_leaf, tree)


def tree_sum(
    tree: PyTree,
    associative_reduction: bool = False,
) -> Tensor:
    """Compute the sum of all elements across all leaves in `tree`.

    Args:
        tree: Input PyTree.
        associative_reduction: Whether to use associative reduction ordering.

    Returns:
        Scalar Tensor holding the sum of all elements.
    """
    del associative_reduction
    total = asarray(0.0)
    for leaf in tree_leaves(tree):
        if leaf is not None:
            total = total + reduce_sum(asarray(leaf))
    return total


def tree_max(tree: PyTree) -> Tensor:
    """Compute the maximum element across all leaves in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Scalar Tensor holding the maximum value (or `-inf` if empty).
    """
    leaves = [asarray(leaf) for leaf in tree_leaves(tree) if leaf is not None]
    if not leaves:
        return asarray(float("-inf"))
    curr_max = reduce_max(leaves[0])
    for leaf in leaves[1:]:
        curr_max = maximum(curr_max, reduce_max(leaf))
    return curr_max


def tree_min(tree: PyTree) -> Tensor:
    """Compute the minimum element across all leaves in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Scalar Tensor holding the minimum value (or `+inf` if empty).
    """
    leaves = [asarray(leaf) for leaf in tree_leaves(tree) if leaf is not None]
    if not leaves:
        return asarray(float("inf"))
    curr_min = reduce_min(leaves[0])
    for leaf in leaves[1:]:
        curr_min = minimum(curr_min, reduce_min(leaf))
    return curr_min


def tree_size(tree: PyTree) -> int:
    """Compute the total number of scalar elements across all leaves in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Total element count across all leaves.
    """
    total = 0
    for leaf in tree_leaves(tree):
        if leaf is not None:
            t = asarray(leaf)
            prod = 1
            for dim in t.shape:
                prod *= dim
            total += prod
    return total


def tree_vdot(tree_x: PyTree, tree_y: PyTree) -> Tensor:
    """Compute the inner product between two PyTrees `tree_x` and `tree_y`.

    Args:
        tree_x: First PyTree.
        tree_y: Second PyTree.

    Returns:
        Scalar Tensor holding the inner product.
    """
    return _base_tree_vdot(tree_x, tree_y)


def tree_l1_norm(tree: PyTree) -> Tensor:
    """Compute the global L1 norm across all leaves in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Scalar Tensor holding the L1 norm.
    """
    total = asarray(0.0)
    for leaf in tree_leaves(tree):
        if leaf is not None:
            total = total + reduce_sum(abs_tensor(asarray(leaf)))
    return total


def tree_l2_norm(tree: PyTree, squared: bool = False) -> Tensor:
    """Compute the global L2 norm (or squared L2 norm) across all leaves in `tree`.

    Args:
        tree: Input PyTree.
        squared: If True, return the squared L2 norm without taking the square root.

    Returns:
        Scalar Tensor holding the L2 norm (or squared L2 norm).
    """
    return _base_tree_l2_norm(tree, squared=squared)


def tree_linf_norm(tree: PyTree) -> Tensor:
    """Compute the global L-infinity norm across all leaves in `tree`.

    Args:
        tree: Input PyTree.

    Returns:
        Scalar Tensor holding the maximum absolute value (0.0 if empty).
    """
    leaves = [
        abs_tensor(asarray(leaf)) for leaf in tree_leaves(tree) if leaf is not None
    ]
    if not leaves:
        return asarray(0.0)
    curr_max = reduce_max(leaves[0])
    for leaf in leaves[1:]:
        curr_max = maximum(curr_max, reduce_max(leaf))
    return curr_max


def tree_norm(
    tree: PyTree,
    ord: int | float | str | None = None,
    squared: bool = False,
) -> Tensor:
    """Compute the vector norm of order `ord` across all leaves in `tree`.

    Args:
        tree: Input PyTree.
        ord: Order of the norm (`None`, `2`, `'fro'`, `1`, `float('inf')`, `'inf'`).
        squared: Whether to return the squared norm.

    Returns:
        Scalar Tensor holding the requested vector norm.

    Raises:
        ValueError: If `ord` is not a supported norm order.
    """
    if ord is None or ord == 2 or ord == "fro":
        return tree_l2_norm(tree, squared=squared)
    if ord == 1:
        n1 = tree_l1_norm(tree)
        return n1 * n1 if squared else n1
    if ord == float("inf") or ord == "inf":
        ninf = tree_linf_norm(tree)
        return ninf * ninf if squared else ninf
    raise ValueError(f"Unsupported norm ord={ord!r}.")


def _structure_signature(tree: PyTree) -> tuple[str, ...]:
    """Build a structural signature tuple for PyTree comparison in `tree_allclose`.

    Args:
        tree: Input PyTree.

    Returns:
        Tuple of structural token strings.
    """
    if isinstance(tree, OptStateProtocol) and not isinstance(tree, tuple):
        dc_tokens: list[str] = [f"dc:{type(tree).__name__}"]
        as_dict = tree._asdict()
        for f in tree._fields:
            dc_tokens.extend(_structure_signature(as_dict[f]))
        return tuple(dc_tokens)
    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        tokens: list[str] = ["map_start"]
        for k in sorted(tree.keys(), key=str):
            tokens.append(f"k:{k}")
            tokens.extend(_structure_signature(_coerce_pytree_child(tree[k])))
        tokens.append("map_end")
        return tuple(tokens)
    if isinstance(tree, NamedTupleProtocol) and isinstance(tree, tuple):
        nt_tokens: list[str] = [f"nt:{type(tree).__name__}"]
        for item in tree:
            nt_tokens.extend(_structure_signature(item))
        return tuple(nt_tokens)
    if isinstance(tree, (tuple, list)):
        seq_tokens: list[str] = [f"seq:{type(tree).__name__}:{len(tree)}"]
        for item in tree:
            seq_tokens.extend(_structure_signature(item))
        return tuple(seq_tokens)
    if tree is None:
        return ("none",)
    return ("leaf",)


def tree_allclose(
    a: PyTree,
    b: PyTree,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """Check whether two PyTrees are element-wise approximately equal within tolerance.

    Args:
        a: First PyTree.
        b: Second PyTree.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether `NaN` values at matching positions compare equal.

    Returns:
        True if structures match and all corresponding leaves are within tolerance.
    """
    if _structure_signature(a) != _structure_signature(b):
        return False
    leaves_a = [la for la in tree_leaves(a) if la is not None]
    leaves_b = [lb for lb in tree_leaves(b) if lb is not None]
    for la, lb in zip(leaves_a, leaves_b):
        ta = asarray(la)
        tb = asarray(lb)
        if ta.shape != tb.shape:
            return False
        list_a = _flatten_tensor_floats(ta)
        list_b = _flatten_tensor_floats(tb)
        for fa, fb in zip(list_a, list_b):
            if math.isnan(fa) or math.isnan(fb):
                if equal_nan and math.isnan(fa) and math.isnan(fb):
                    continue
                return False
            if abs(fa - fb) > atol + rtol * abs(fb):
                return False
    return True


def tree_update_moment(
    updates: Updates,
    moments: OptState,
    decay: float,
    order: int,
) -> OptState:
    """Compute the exponential moving average of the `order`-th moment.

    Args:
        updates: Gradient updates PyTree.
        moments: Current moment accumulator PyTree.
        decay: Exponential decay factor in `[0, 1]`.
        order: Moment exponent order (e.g., `1` or `2`).

    Returns:
        Updated moment PyTree `(1 - decay) * (updates ** order) + decay * moments`.
    """

    def _step(g: TensorLeaf, m: TensorLeaf) -> TensorLeaf:
        """Update single moment leaf.

        Args:
            g: Gradient leaf.
            m: Moment leaf.

        Returns:
            Updated moment leaf.
        """
        if g is None or m is None:
            return None
        g_t = asarray(g)
        m_t = asarray(m)
        return (1.0 - decay) * (g_t**order) + decay * m_t

    return tree_map(_step, updates, moments)


def tree_update_moment_per_elem_norm(
    updates: Updates,
    moments: OptState,
    decay: float,
    order: int,
) -> OptState:
    """Compute the EMA of the `order`-th moment of the element-wise norm.

    Args:
        updates: Gradient updates PyTree.
        moments: Current moment accumulator PyTree.
        decay: Exponential decay factor in `[0, 1]`.
        order: Moment exponent order.

    Returns:
        Updated moment PyTree `(1 - decay) * (|updates| ** order) + decay * moments`.
    """

    def _step(g: TensorLeaf, m: TensorLeaf) -> TensorLeaf:
        """Update single element-wise norm moment leaf.

        Args:
            g: Gradient leaf.
            m: Moment leaf.

        Returns:
            Updated moment leaf.
        """
        if g is None or m is None:
            return None
        g_abs = abs_tensor(asarray(g))
        m_t = asarray(m)
        return (1.0 - decay) * (g_abs**order) + decay * m_t

    return tree_map(_step, updates, moments)


def tree_update_infinity_moment(
    updates: Updates,
    moments: OptState,
    decay: float,
    eps: float,
) -> OptState:
    """Compute the exponential moving average of the infinity norm (`max(decay * m, |g| + eps)`).

    Args:
        updates: Gradient updates PyTree.
        moments: Current infinity-norm moment accumulator PyTree.
        decay: Exponential decay factor.
        eps: Non-negative constant added to `|g|` for numerical stability.

    Returns:
        Updated infinity-norm moment PyTree.
    """

    def _step(g: TensorLeaf, m: TensorLeaf) -> TensorLeaf:
        """Update single infinity-moment leaf.

        Args:
            g: Gradient leaf.
            m: Infinity-moment leaf.

        Returns:
            Updated infinity-moment leaf.
        """
        if g is None or m is None:
            return g
        g_t = asarray(g)
        m_t = asarray(m)
        return maximum(abs_tensor(g_t) + eps, decay * m_t)

    return tree_map(_step, updates, moments)


def tree_bias_correction(
    moment: OptState,
    decay: float,
    count: int | Tensor,
) -> OptState:
    """Perform bias correction `moment / (1 - decay ** count)` across `moment`.

    Args:
        moment: Moment accumulator PyTree.
        decay: Exponential decay factor.
        count: Optimization step count.

    Returns:
        Bias-corrected moment PyTree.
    """
    count_t = asarray(count, dtype="float32")
    denom = 1.0 - power(asarray(decay, dtype="float32"), count_t)
    safe_denom = where(abs_tensor(denom) > 1e-12, denom, 1.0)

    def _div_bc(m: TensorLeaf) -> TensorLeaf:
        """Divide moment leaf by `1 - decay ** count`.

        Args:
            m: Moment leaf.

        Returns:
            Bias-corrected leaf.
        """
        if m is None:
            return None
        m_t = asarray(m)
        return asarray(m_t / safe_denom, dtype=m_t.dtype)

    return tree_map(_div_bc, moment)


__all__ = [
    "_flatten_tensor_floats",
    "tree_add",
    "tree_add_scale",
    "tree_allclose",
    "tree_batch_shape",
    "tree_bias_correction",
    "tree_cast",
    "tree_cast_like",
    "tree_clip",
    "tree_conj",
    "tree_div",
    "tree_dtype",
    "tree_full_like",
    "tree_l1_norm",
    "tree_l2_norm",
    "tree_linf_norm",
    "tree_max",
    "tree_min",
    "tree_mul",
    "tree_norm",
    "tree_ones_like",
    "tree_real",
    "tree_scalar_mul",
    "tree_scale",
    "tree_size",
    "tree_sub",
    "tree_sum",
    "tree_update_infinity_moment",
    "tree_update_moment",
    "tree_update_moment_per_elem_norm",
    "tree_vdot",
    "tree_where",
    "tree_zeros_like",
]
