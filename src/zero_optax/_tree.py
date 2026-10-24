"""PyTree traversal and typed tensor math helpers built on ml-switcheroo-compiler."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.shape import indexing
from ml_switcheroo_compiler.ops.shape import reshape as shape_reshape

from zero_optax._types import (
    ArrayLike,
    DTypeLike,
    MaskOrFn,
    MaskTree,
    Params,
    PyTree,
    ScalarOrSchedule,
    TensorLeaf,
    Updates,
)

INF: float = float("inf")
NEG_INF: float = float("-inf")


def _resolve_dtype(dtype: DTypeLike) -> DType | None:
    """Resolve a `DTypeLike` value into an `ml_switcheroo_ir` `DType` enum or None.

    Args:
        dtype: Optional `DType`, string name, or Python numeric type.

    Returns:
        Resolved `DType` instance or None.
    """
    if dtype is None or isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        return DType(dtype)
    if dtype is bool:
        return DType("bool")
    if dtype is float:
        return DType.float32
    return DType.int32


def asarray(val: ArrayLike, dtype: DTypeLike = None) -> Tensor:
    """Convert an ArrayLike value into an ml-switcheroo-compiler Tensor.

    Args:
        val: Input tensor, scalar, or nested sequence of numbers.
        dtype: Optional target data type.

    Returns:
        Converted Tensor instance.
    """
    if isinstance(val, Tensor) and dtype is None:
        return val
    return as_tensor_obj(ops.asarray(val, dtype=_resolve_dtype(dtype)))


def zeros(shape: int | Sequence[int], dtype: DTypeLike = None) -> Tensor:
    """Create a Tensor filled with zeros.

    Args:
        shape: Integer dimension or sequence of dimensions.
        dtype: Optional target data type.

    Returns:
        Zero-initialized Tensor of the requested shape.
    """
    norm_shape: tuple[int, ...] = (shape,) if isinstance(shape, int) else tuple(shape)
    return as_tensor_obj(ops.zeros(norm_shape, dtype=_resolve_dtype(dtype)))


def zeros_like(x: ArrayLike, dtype: DTypeLike = None) -> Tensor:
    """Create a Tensor of zeros with the same shape and dtype as the input.

    Args:
        x: Reference array or tensor.
        dtype: Optional override data type.

    Returns:
        Zero-initialized Tensor matching the shape of `x`.
    """
    t = asarray(x)
    target_dtype: DTypeLike = t.dtype if dtype is None else dtype
    return as_tensor_obj(ops.zeros_like(t, dtype=_resolve_dtype(target_dtype)))


def ones(shape: int | Sequence[int], dtype: DTypeLike = None) -> Tensor:
    """Create a Tensor filled with ones.

    Args:
        shape: Integer dimension or sequence of dimensions.
        dtype: Optional target data type.

    Returns:
        One-initialized Tensor of the requested shape.
    """
    norm_shape: tuple[int, ...] = (shape,) if isinstance(shape, int) else tuple(shape)
    return as_tensor_obj(ops.ones(norm_shape, dtype=_resolve_dtype(dtype)))


def ones_like(x: ArrayLike, dtype: DTypeLike = None) -> Tensor:
    """Create a Tensor of ones with the same shape and dtype as the input.

    Args:
        x: Reference array or tensor.
        dtype: Optional override data type.

    Returns:
        One-initialized Tensor matching the shape of `x`.
    """
    t = asarray(x)
    target_dtype: DTypeLike = t.dtype if dtype is None else dtype
    return as_tensor_obj(ops.ones_like(t, dtype=_resolve_dtype(target_dtype)))


def full_like(x: ArrayLike, fill_value: float | int, dtype: DTypeLike = None) -> Tensor:
    """Create a Tensor filled with a scalar value matching the shape of `x`.

    Args:
        x: Reference array or tensor.
        fill_value: Scalar constant to fill the tensor with.
        dtype: Optional override data type.

    Returns:
        Constant-filled Tensor matching the shape of `x`.
    """
    t = asarray(x)
    target_dtype: DTypeLike = t.dtype if dtype is None else dtype
    return as_tensor_obj(
        ops.full_like(t, fill_value, dtype=_resolve_dtype(target_dtype))
    )


def arange(start: int, stop: int | None = None, step: int = 1) -> Tensor:
    """Create a 1D Tensor of evenly spaced values within a given interval.

    Args:
        start: Start of interval (or stop if `stop` is None).
        stop: End of interval.
        step: Spacing between values.

    Returns:
        1D Tensor of sequential values.
    """
    if stop is None:
        return as_tensor_obj(ops.arange(0, start, step))
    return as_tensor_obj(ops.arange(start, stop, step))


def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> Tensor:
    """Select elements from `x` or `y` depending on `condition`.

    Args:
        condition: Boolean mask tensor or scalar.
        x: Values selected where `condition` is True.
        y: Values selected where `condition` is False.

    Returns:
        Tensor with selected elements.
    """
    return as_tensor_obj(indexing.where(asarray(condition), asarray(x), asarray(y)))


def maximum(x: ArrayLike, y: ArrayLike) -> Tensor:
    """Compute element-wise maximum of two array-like inputs.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Element-wise maximum Tensor.
    """
    return as_tensor_obj(ops.maximum(asarray(x), asarray(y)))


def minimum(x: ArrayLike, y: ArrayLike) -> Tensor:
    """Compute element-wise minimum of two array-like inputs.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Element-wise minimum Tensor.
    """
    return as_tensor_obj(ops.minimum(asarray(x), asarray(y)))


def abs_tensor(x: ArrayLike) -> Tensor:
    """Compute element-wise absolute value.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise absolute value Tensor.
    """
    return as_tensor_obj(ops.abs(asarray(x)))


def exp(x: ArrayLike) -> Tensor:
    """Compute element-wise exponential.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise exponential Tensor.
    """
    return as_tensor_obj(ops.exp(asarray(x)))


def log(x: ArrayLike) -> Tensor:
    """Compute element-wise natural logarithm.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise natural logarithm Tensor.
    """
    return as_tensor_obj(ops.log(asarray(x)))


def log1p(x: ArrayLike) -> Tensor:
    """Compute element-wise log(1 + x) accurately for small x.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise log(1 + x) Tensor.
    """
    return as_tensor_obj(ops.log1p(asarray(x)))


def sqrt(x: ArrayLike) -> Tensor:
    """Compute element-wise non-negative square root.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise square root Tensor.
    """
    return as_tensor_obj(ops.sqrt(asarray(x)))


def sign(x: ArrayLike) -> Tensor:
    """Compute element-wise sign indication (-1, 0, +1).

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise sign Tensor.
    """
    return as_tensor_obj(ops.sign(asarray(x)))


def sigmoid(x: ArrayLike) -> Tensor:
    """Compute numerically stable element-wise logistic sigmoid.

    Args:
        x: Input tensor or scalar.

    Returns:
        Sigmoid probability Tensor in [0, 1].
    """
    t = asarray(x)
    neg_abs = -abs_tensor(t)
    exp_neg = exp(neg_abs)
    denom = 1.0 + exp_neg
    pos_sig = 1.0 / denom
    neg_sig = exp_neg / denom
    return where(t >= 0.0, pos_sig, neg_sig)


def reduce_sum(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute sum of tensor elements over a given axis or axes.

    Args:
        x: Input tensor or array-like.
        axis: Axis or tuple of axes along which a sum is performed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced sum Tensor.
    """
    return as_tensor_obj(ops.sum(asarray(x), axis=axis, keepdims=keepdims))


def reduce_mean(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute arithmetic mean of tensor elements over a given axis or axes.

    Args:
        x: Input tensor or array-like.
        axis: Axis or tuple of axes along which the mean is computed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced mean Tensor.
    """
    return as_tensor_obj(ops.mean(asarray(x), axis=axis, keepdims=keepdims))


def reduce_max(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute maximum of tensor elements over a given axis or axes.

    Args:
        x: Input tensor or array-like.
        axis: Axis or tuple of axes along which the maximum is computed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced maximum Tensor.
    """
    return as_tensor_obj(ops.max(asarray(x), axis=axis, keepdims=keepdims))


def expand_dims(x: ArrayLike, axis: int) -> Tensor:
    """Expand the shape of a tensor by inserting a new axis of length 1.

    Args:
        x: Input tensor or array-like.
        axis: Position in the expanded axes where the new axis is placed.

    Returns:
        Tensor with expanded dimensions.
    """
    return as_tensor_obj(ops.expand_dims(asarray(x), axis=axis))


def squeeze(x: ArrayLike, axis: int | tuple[int, ...] | None = None) -> Tensor:
    """Remove axes of length one from `x`.

    Args:
        x: Input tensor or array-like.
        axis: Selects a subset of the entries of length one in the shape.

    Returns:
        Squeezed Tensor.
    """
    return as_tensor_obj(ops.squeeze(asarray(x), axis=axis))


def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis: int) -> Tensor:
    """Take values from the input tensor by matching 1D index and data slices.

    Args:
        arr: Source tensor.
        indices: Indices to take along each 1D slice of `arr`.
        axis: The axis to take 1D slices along.

    Returns:
        Indexed result wrapped as a Tensor.
    """
    raw = indexing.take_along_axis(asarray(arr), asarray(indices), axis=axis)
    return asarray(as_tensor_obj(raw))


def reshape(x: ArrayLike, newshape: Sequence[int]) -> Tensor:
    """Give a new shape to a tensor without changing its data.

    Args:
        x: Input tensor to be reshaped.
        newshape: New shape tuple or sequence of ints.

    Returns:
        Reshaped Tensor.
    """
    return as_tensor_obj(shape_reshape.Reshape()(asarray(x), tuple(newshape)))


LeafFn1 = Callable[[TensorLeaf], TensorLeaf]
LeafFn2 = Callable[[TensorLeaf, TensorLeaf], TensorLeaf]
LeafFn3 = Callable[[TensorLeaf, TensorLeaf, TensorLeaf], TensorLeaf]
LeafFn4 = Callable[[TensorLeaf, TensorLeaf, TensorLeaf, TensorLeaf], TensorLeaf]


def _apply_leaf_fn(
    fn: LeafFn1 | LeafFn2 | LeafFn3 | LeafFn4,
    first: TensorLeaf,
    rest_leaves: Sequence[TensorLeaf],
) -> TensorLeaf:
    """Dispatch `fn` with the exact arity matching `1 + len(rest_leaves)`.

    Args:
        fn: Leaf transformation callable taking 1 to 4 arguments.
        first: Primary leaf value.
        rest_leaves: Additional leaf values from co-traversed PyTrees.

    Returns:
        Transformed `TensorLeaf` value.
    """
    n_extra = len(rest_leaves)
    if n_extra == 0:
        fn1: LeafFn1 = fn
        return fn1(first)
    if n_extra == 1:
        fn2: LeafFn2 = fn
        return fn2(first, rest_leaves[0])
    if n_extra == 2:
        fn3: LeafFn3 = fn
        return fn3(first, rest_leaves[0], rest_leaves[1])
    fn4: LeafFn4 = fn
    return fn4(first, rest_leaves[0], rest_leaves[1], rest_leaves[2])


def tree_map(
    fn: LeafFn1 | LeafFn2 | LeafFn3 | LeafFn4,
    tree: PyTree,
    *rest: PyTree,
) -> PyTree:
    """Apply a function to every leaf across one or more PyTrees with matching structure.

    Args:
        fn: Callable taking 1 to 4 leaves (one per input tree) and returning a transformed leaf.
        tree: Primary PyTree structure.
        *rest: Additional PyTrees with identical container structure.

    Returns:
        New PyTree with the same structure as `tree` and transformed leaves.

    Raises:
        ValueError: If any PyTree in `rest` has a structure or length mismatch with `tree`.
    """
    if isinstance(tree, Mapping):
        result_dict: dict[str, PyTree] = {}
        tree_keys = set(tree.keys())
        for r in rest:
            if not isinstance(r, Mapping) or set(r.keys()) != tree_keys:
                raise ValueError(
                    "PyTree structure mismatch: expected Mapping with matching keys."
                )
        for k, v in tree.items():
            other_vals: list[PyTree] = [r[k] for r in rest if isinstance(r, Mapping)]
            result_dict[k] = tree_map(fn, v, *other_vals)
        return result_dict
    if isinstance(tree, (tuple, list)):
        seq_items: list[PyTree] = []
        tree_len = len(tree)
        rest_seqs: list[tuple[PyTree, ...] | list[PyTree]] = []
        for r in rest:
            if not isinstance(r, (tuple, list)) or len(r) != tree_len:
                raise ValueError(
                    "PyTree structure mismatch: expected sequence of matching length."
                )
            rest_seqs.append(r)
        for idx, item in enumerate(tree):
            other_items: list[PyTree] = [r_seq[idx] for r_seq in rest_seqs]
            seq_items.append(tree_map(fn, item, *other_items))
        return list(seq_items) if isinstance(tree, list) else tuple(seq_items)
    for r in rest:
        if isinstance(r, (Mapping, tuple, list)):
            raise ValueError(
                "PyTree structure mismatch: expected leaf node across all trees."
            )
    leaf_rest: list[TensorLeaf] = [
        r for r in rest if not isinstance(r, (Mapping, tuple, list))
    ]
    return _apply_leaf_fn(fn, tree, leaf_rest)


def tree_leaves(tree: PyTree) -> list[TensorLeaf]:
    """Collect all leaf values from a PyTree in depth-first order.

    Args:
        tree: Input PyTree structure.

    Returns:
        Flat list of TensorLeaf values.
    """
    if isinstance(tree, Mapping):
        leaves: list[TensorLeaf] = []
        for v in tree.values():
            leaves.extend(tree_leaves(v))
        return leaves
    if isinstance(tree, (tuple, list)):
        seq_leaves: list[TensorLeaf] = []
        for item in tree:
            seq_leaves.extend(tree_leaves(item))
        return seq_leaves
    return [tree]


def tree_zeros_like(tree: Params, dtype: DTypeLike = None) -> Updates:
    """Create an identical PyTree structure where every leaf is a zero Tensor.

    Args:
        tree: Input parameter PyTree.
        dtype: Optional override data type for zero tensors.

    Returns:
        Zero-initialized PyTree matching `tree`.
    """

    def _zero_leaf(x: TensorLeaf) -> TensorLeaf:
        """Create a zero Tensor matching leaf `x`.

        Args:
            x: Leaf value from parameter PyTree.

        Returns:
            Zero Tensor matching `x`.
        """
        if x is None:
            return None
        return zeros_like(x, dtype=dtype)

    return tree_map(_zero_leaf, tree)


def tree_l2_norm(tree: PyTree, squared: bool = False) -> Tensor:
    """Compute the global L2 norm across all leaves in a PyTree.

    Args:
        tree: Input PyTree of tensors or scalars.
        squared: If True, return the sum of squared elements without taking the square root.

    Returns:
        Scalar Tensor holding the L2 norm (or squared L2 norm).
    """
    leaves = tree_leaves(tree)
    total_sq: Tensor = asarray(0.0)
    for leaf in leaves:
        if leaf is not None:
            t = asarray(leaf)
            total_sq = total_sq + reduce_sum(t * t)
    if squared:
        return total_sq
    return sqrt(total_sq)


def leaf_l2_norm(x: TensorLeaf) -> Tensor:
    """Compute the L2 norm of a single leaf tensor.

    Args:
        x: Input leaf tensor or scalar.

    Returns:
        Scalar Tensor holding the L2 norm of `x`.
    """
    if x is None:
        return asarray(0.0)
    t = asarray(x)
    return sqrt(reduce_sum(t * t))


def evaluate_schedule(
    lr_or_schedule: ScalarOrSchedule, count: Tensor | int
) -> float | Tensor:
    """Evaluate a learning rate or weight decay scalar/schedule at step `count`.

    Args:
        lr_or_schedule: Scalar float/int/Tensor or callable schedule function.
        count: Current integer or Tensor step count.

    Returns:
        Evaluated scalar or Tensor step value.
    """
    if callable(lr_or_schedule):
        return lr_or_schedule(count)
    return lr_or_schedule


def resolve_mask(mask: MaskOrFn, params: Params) -> MaskTree | None:
    """Resolve an optional mask or mask-producing callable against `params`.

    Args:
        mask: Boolean mask PyTree, callable `(Params) -> MaskTree`, or None.
        params: Current model parameters PyTree.

    Returns:
        Resolved MaskTree or None.
    """
    if mask is None:
        return None
    if callable(mask):
        return mask(params)
    return mask


def apply_mask_tree(
    masked_tree: Updates,
    unmasked_tree: Updates,
    mask_tree: MaskTree,
) -> Updates:
    """Select leaves from `masked_tree` where `mask_tree` is True and `unmasked_tree` where False.

    Args:
        masked_tree: PyTree of updates with transformation applied.
        unmasked_tree: PyTree of fallback updates when masked out.
        mask_tree: Boolean or nested structure of booleans matching the PyTree.

    Returns:
        Combined PyTree respecting the mask.
    """
    if isinstance(mask_tree, bool):
        return masked_tree if mask_tree else unmasked_tree
    if (
        isinstance(mask_tree, Mapping)
        and isinstance(masked_tree, Mapping)
        and isinstance(unmasked_tree, Mapping)
    ):
        return {
            k: apply_mask_tree(masked_tree[k], unmasked_tree[k], mask_tree.get(k, True))
            for k in masked_tree
        }
    if (
        isinstance(mask_tree, tuple)
        and isinstance(masked_tree, tuple)
        and isinstance(unmasked_tree, tuple)
    ):
        return tuple(
            apply_mask_tree(m_val, u_val, b_val)
            for m_val, u_val, b_val in zip(masked_tree, unmasked_tree, mask_tree)
        )
    if (
        isinstance(mask_tree, list)
        and isinstance(masked_tree, list)
        and isinstance(unmasked_tree, list)
    ):
        return [
            apply_mask_tree(m_val, u_val, b_val)
            for m_val, u_val, b_val in zip(masked_tree, unmasked_tree, mask_tree)
        ]
    return masked_tree


def safe_norm(
    x: ArrayLike,
    min_norm: float | ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the L2 norm of `x` bounded below by `min_norm` for safe gradient division.

    Args:
        x: Input tensor.
        min_norm: Minimum lower bound on the returned norm.
        axis: Axis or axes along which to compute the norm.
        keepdims: Whether to keep reduced dimensions as length 1.

    Returns:
        Tensor of safe norms.
    """
    t = asarray(x)
    min_t = asarray(min_norm)
    sq_sum = reduce_sum(t * t, axis=axis, keepdims=keepdims)
    norm_val = sqrt(maximum(sq_sum, min_t * min_t))
    return maximum(norm_val, min_t)


def logsumexp(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    keepdims: bool = False,
) -> Tensor:
    """Compute numerically stable log(sum(exp(x))) along `axis`.

    Args:
        x: Input log-potentials tensor.
        axis: Axis or axes to reduce over.
        keepdims: Whether to retain reduced dimensions.

    Returns:
        Numerically stable log-sum-exp Tensor.
    """
    t = asarray(x)
    c = reduce_max(t, axis=axis, keepdims=True)
    c_safe = where(abs_tensor(c) == INF, 0.0, c)
    shifted = where(abs_tensor(t) == INF, NEG_INF, t - c_safe)
    sum_exp = reduce_sum(exp(shifted), axis=axis, keepdims=keepdims)
    c_out = c_safe if keepdims else squeeze_axes(c_safe, axis)
    return c_out + log(where(sum_exp == 0.0, 1.0, sum_exp))


def squeeze_axes(x: Tensor, axis: int | tuple[int, ...] | None) -> Tensor:
    """Remove specified singleton axes from a tensor after a `keepdims=True` reduction.

    Args:
        x: Tensor with singleton dimensions at `axis`.
        axis: Axis or axes to remove, or None to squeeze all dimensions to a scalar.

    Returns:
        Reshaped Tensor with the target axes removed.
    """
    if axis is None:
        return reshape(x, ())
    ndim = x.ndim
    axes = (axis,) if isinstance(axis, int) else axis
    norm_axes = {a if a >= 0 else ndim + a for a in axes}
    new_shape = [dim for i, dim in enumerate(x.shape) if i not in norm_axes]
    return reshape(x, new_shape)


__all__ = [
    "INF",
    "NEG_INF",
    "abs_tensor",
    "apply_mask_tree",
    "arange",
    "asarray",
    "evaluate_schedule",
    "exp",
    "expand_dims",
    "full_like",
    "leaf_l2_norm",
    "log",
    "log1p",
    "logsumexp",
    "maximum",
    "minimum",
    "ones",
    "ones_like",
    "reduce_max",
    "reduce_mean",
    "reduce_sum",
    "reshape",
    "resolve_mask",
    "safe_norm",
    "sigmoid",
    "sign",
    "sqrt",
    "squeeze",
    "squeeze_axes",
    "take_along_axis",
    "tree_l2_norm",
    "tree_leaves",
    "tree_map",
    "tree_zeros_like",
    "where",
    "zeros",
    "zeros_like",
]
