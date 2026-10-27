"""PyTree traversal and typed tensor math helpers built on zero-jax and ml-switcheroo-compiler."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Union

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.ops.shape import indexing
from ml_switcheroo_compiler.ops.shape import reshape as shape_reshape
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import (
    get_vjp_from_data,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    UnconnectedGradients,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from typing_extensions import TypeGuard
from zero_flax.nnx.filterlib import Filter, to_predicate
from zero_flax.nnx.variables import Variable, VariableBase
from zero_jax import lax
from zero_jax import tree_util as jax_tree_util
from zero_jax._types import ArrayLikeSpec as JaxArrayLikeSpec
from zero_jax.numpy import ndarray as JaxArray
from zero_jax.numpy._ndarray import _to_tensor as _jax_to_tensor
from zero_orbax.types import PyTree as OrbaxPyTree

from zero_optax._types import (
    ArrayLike,
    ArrayLikeProtocol,
    CustomMappingChild,
    CustomMappingPyTreeProtocol,
    DTypeLike,
    DTypeLikeProtocol,
    MappableDataclassProtocol,
    MaskedNode,
    MaskOrFn,
    MaskTree,
    NamedTupleProtocol,
    Params,
    PyTree,
    Scalar,
    ScalarOrSchedule,
    ShapeDtypeProtocol,
    TensorLeaf,
    ToListArrayProtocol,
    Updates,
    VariableLeafProtocol,
)

INF: float = float("inf")
NEG_INF: float = float("-inf")


def _extract_node_shape(
    graph: LogicalGraph,
    node_id: str,
) -> tuple[int, ...]:
    """Extract static tuple shape metadata from a node in `graph`.

    Args:
        graph: Active logical IR graph.
        node_id: Identifier of the node to inspect.

    Returns:
        Tuple of integer dimensions.
    """
    raw_shape = graph.nodes[node_id].shape_metadata
    return (
        tuple(int(d) for d in raw_shape if isinstance(d, int))
        if isinstance(raw_shape, tuple)
        else ()
    )


def _expand_reduced_cotangent(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
    inp_shape: tuple[int, ...],
) -> tuple[str, int]:
    """Expand reduced axes on `cotangent` and compute the number of reduced elements.

    Args:
        graph: Active backward IR graph.
        node: Forward reduction `LogicalNode` (`Sum`, `Mean`, `Max`, `Min`).
        cotangent: Incoming cotangent node ID.
        inp_shape: Static shape of the forward reduction input tensor.

    Returns:
        Tuple `(expanded_cotangent_id, num_reduced_elements)`.
    """
    axis_attr = node.attributes.get("axis")
    keepdims = bool(node.attributes.get("keepdims", False))
    cur = cotangent
    num_reduced = 1
    ndim = len(inp_shape)
    if axis_attr is None:
        for dim in inp_shape:
            num_reduced *= max(1, dim)
    else:
        ax_int = int(axis_attr) if isinstance(axis_attr, int) else 0
        ax = ax_int if ax_int >= 0 else ndim + ax_int
        num_reduced *= max(1, inp_shape[ax])
        if not keepdims:
            cur = emit_ir_node(
                graph,
                "ExpandDims",
                [cur],
                shape_metadata=inp_shape,
                attributes={"axis": ax},
            )
    return cur, max(1, num_reduced)


def _sum_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str]:
    """Compute the VJP of `Sum` / `ReduceSum` with axis-aware dimension expansion.

    Args:
        graph: Active backward IR graph.
        node: Forward `Sum` or `ReduceSum` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Single-element tuple containing the broadcasted gradient node ID.
    """
    inp_id = node.inputs[0]
    inp_shape = _extract_node_shape(graph, inp_id)
    cur, _ = _expand_reduced_cotangent(graph, node, cotangent, inp_shape)
    bcast = emit_ir_node(
        graph,
        "BroadcastLike",
        [cur, inp_id],
        shape_metadata=inp_shape,
        attributes={"target_id": inp_id},
    )
    return (bcast,)


def _mean_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str]:
    """Compute the VJP of `Mean` / `ReduceMean` dividing by the reduced element count.

    Args:
        graph: Active backward IR graph.
        node: Forward `Mean` or `ReduceMean` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Single-element tuple containing the scaled broadcasted gradient node ID.
    """
    inp_id = node.inputs[0]
    inp_shape = _extract_node_shape(graph, inp_id)
    cur, count = _expand_reduced_cotangent(graph, node, cotangent, inp_shape)
    bcast = emit_ir_node(
        graph,
        "BroadcastLike",
        [cur, inp_id],
        shape_metadata=inp_shape,
        attributes={"target_id": inp_id},
    )
    denom = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": float(count)},
    )
    scaled = emit_ir_node(
        graph,
        "Divide",
        [bcast, denom],
        shape_metadata=inp_shape,
        attributes={},
    )
    return (scaled,)


def _extrema_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str]:
    """Compute the VJP of `Max` / `ReduceMax` / `Min` / `ReduceMin` splitting ties evenly.

    Args:
        graph: Active backward IR graph.
        node: Forward extremum reduction node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Single-element tuple containing the extremum-routed gradient node ID.
    """
    inp_id = node.inputs[0]
    inp_shape = _extract_node_shape(graph, inp_id)
    cur_cot, _ = _expand_reduced_cotangent(graph, node, cotangent, inp_shape)
    cur_out, _ = _expand_reduced_cotangent(graph, node, node.id, inp_shape)
    bcast_cot = emit_ir_node(
        graph,
        "BroadcastLike",
        [cur_cot, inp_id],
        shape_metadata=inp_shape,
        attributes={"target_id": inp_id},
    )
    bcast_out = emit_ir_node(
        graph,
        "BroadcastLike",
        [cur_out, inp_id],
        shape_metadata=inp_shape,
        attributes={"target_id": inp_id},
    )
    eq_mask = emit_ir_node(
        graph,
        "Equal",
        [inp_id, bcast_out],
        shape_metadata=inp_shape,
        attributes={},
    )
    zero_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 0.0},
    )
    routed = emit_ir_node(
        graph,
        "Where",
        [eq_mask, bcast_cot, zero_c],
        shape_metadata=inp_shape,
        attributes={},
    )
    return (routed,)


def _bcast_reduce_to_input(
    graph: LogicalGraph,
    adj_id: str,
    inp_id: str,
) -> str:
    """Reduce broadcasted axes of `adj_id` to match the shape of `inp_id`.

    Args:
        graph: Active backward IR graph.
        adj_id: Raw gradient node ID.
        inp_id: Forward input node ID whose shape should be matched.

    Returns:
        Shape-aligned gradient node ID.
    """
    inp_shape = _extract_node_shape(graph, inp_id)
    node_id: str = str(
        emit_ir_node(
            graph,
            "BroadcastReduce",
            [adj_id, inp_id],
            shape_metadata=inp_shape,
            attributes={"target_id": inp_id},
        )
    )
    return node_id


def _where_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[UnconnectedGradients, str, str]:
    """Compute the VJP of `Where(condition, x, y)` wrt `condition`, `x`, and `y`.

    Args:
        graph: Active backward IR graph.
        node: Forward `Where` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple `(ZERO, grad_x, grad_y)`.
    """
    cond_id, x_id, y_id = node.inputs[0], node.inputs[1], node.inputs[2]
    out_shape = _extract_node_shape(graph, node.id)
    zero_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 0.0},
    )
    gx_raw = emit_ir_node(
        graph,
        "Where",
        [cond_id, cotangent, zero_c],
        shape_metadata=out_shape,
        attributes={},
    )
    gy_raw = emit_ir_node(
        graph,
        "Where",
        [cond_id, zero_c, cotangent],
        shape_metadata=out_shape,
        attributes={},
    )
    return (
        UnconnectedGradients.ZERO,
        _bcast_reduce_to_input(graph, gx_raw, x_id),
        _bcast_reduce_to_input(graph, gy_raw, y_id),
    )


def _maximum_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str, str]:
    """Compute the VJP of `Maximum(x, y)` with broadcast reduction on both inputs.

    Args:
        graph: Active backward IR graph.
        node: Forward `Maximum` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple `(grad_x, grad_y)`.
    """
    x_id, y_id = node.inputs[0], node.inputs[1]
    out_shape = _extract_node_shape(graph, node.id)
    cond = emit_ir_node(
        graph,
        "GreaterEqual",
        [x_id, y_id],
        shape_metadata=out_shape,
        attributes={},
    )
    zero_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 0.0},
    )
    gx = emit_ir_node(
        graph,
        "Where",
        [cond, cotangent, zero_c],
        shape_metadata=out_shape,
        attributes={},
    )
    gy = emit_ir_node(
        graph,
        "Where",
        [cond, zero_c, cotangent],
        shape_metadata=out_shape,
        attributes={},
    )
    return (
        _bcast_reduce_to_input(graph, gx, x_id),
        _bcast_reduce_to_input(graph, gy, y_id),
    )


def _minimum_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str, str]:
    """Compute the VJP of `Minimum(x, y)` with broadcast reduction on both inputs.

    Args:
        graph: Active backward IR graph.
        node: Forward `Minimum` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple `(grad_x, grad_y)`.
    """
    x_id, y_id = node.inputs[0], node.inputs[1]
    out_shape = _extract_node_shape(graph, node.id)
    cond = emit_ir_node(
        graph,
        "LessEqual",
        [x_id, y_id],
        shape_metadata=out_shape,
        attributes={},
    )
    zero_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 0.0},
    )
    gx = emit_ir_node(
        graph,
        "Where",
        [cond, cotangent, zero_c],
        shape_metadata=out_shape,
        attributes={},
    )
    gy = emit_ir_node(
        graph,
        "Where",
        [cond, zero_c, cotangent],
        shape_metadata=out_shape,
        attributes={},
    )
    return (
        _bcast_reduce_to_input(graph, gx, x_id),
        _bcast_reduce_to_input(graph, gy, y_id),
    )


def _power_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str, str | UnconnectedGradients]:
    """Compute the VJP of `Power(x, y)` / `Pow(x, y)` avoiding `log(0)` when `y` is constant.

    Args:
        graph: Active backward IR graph.
        node: Forward `Power` or `Pow` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple `(grad_x, grad_y)`.
    """
    x_id, y_id = node.inputs[0], node.inputs[1]
    out_shape = _extract_node_shape(graph, node.id)
    one_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 1.0},
    )
    y_minus_1 = emit_ir_node(
        graph,
        "Subtract",
        [y_id, one_c],
        shape_metadata=out_shape,
        attributes={},
    )
    x_pow = emit_ir_node(
        graph,
        "Power",
        [x_id, y_minus_1],
        shape_metadata=out_shape,
        attributes={},
    )
    y_x_pow = emit_ir_node(
        graph,
        "Multiply",
        [y_id, x_pow],
        shape_metadata=out_shape,
        attributes={},
    )
    gx_raw = emit_ir_node(
        graph,
        "Multiply",
        [cotangent, y_x_pow],
        shape_metadata=out_shape,
        attributes={},
    )
    gx = _bcast_reduce_to_input(graph, gx_raw, x_id)
    y_node = graph.nodes.get(y_id)
    if y_node is not None and y_node.op_type == "Constant":
        return (gx, UnconnectedGradients.ZERO)
    eps_c = emit_ir_node(
        graph,
        "Constant",
        [],
        shape_metadata=(),
        attributes={"value": 1e-30},
    )
    safe_x = emit_ir_node(
        graph,
        "Maximum",
        [x_id, eps_c],
        shape_metadata=out_shape,
        attributes={},
    )
    log_x = emit_ir_node(
        graph,
        "Log",
        [safe_x],
        shape_metadata=out_shape,
        attributes={},
    )
    x_pow_y = emit_ir_node(
        graph,
        "Power",
        [x_id, y_id],
        shape_metadata=out_shape,
        attributes={},
    )
    log_term = emit_ir_node(
        graph,
        "Multiply",
        [log_x, x_pow_y],
        shape_metadata=out_shape,
        attributes={},
    )
    gy_raw = emit_ir_node(
        graph,
        "Multiply",
        [cotangent, log_term],
        shape_metadata=out_shape,
        attributes={},
    )
    return (gx, _bcast_reduce_to_input(graph, gy_raw, y_id))


def _make_bcast_binary_vjp(
    op_name: str,
) -> Callable[[LogicalGraph, LogicalNode, str], tuple[str, ...]]:
    """Wrap a declarative binary VJP rule with `BroadcastReduce` on each input.

    Args:
        op_name: Canonical binary operation name (`Add`, `Subtract`, `Multiply`, `Divide`, etc.).

    Returns:
        VJP rule callable that reduces broadcasted dimensions to match each input shape.
    """
    base_vjp = get_vjp_from_data(op_name)

    def _wrapped(
        graph: LogicalGraph,
        node: LogicalNode,
        cotangent: str,
    ) -> tuple[str, ...]:
        """Apply `base_vjp` and reduce broadcasted dimensions to match `node.inputs`.

        Args:
            graph: Active backward IR graph.
            node: Forward binary operation node.
            cotangent: Incoming cotangent node ID.

        Returns:
            Tuple of shape-aligned input gradient node IDs.
        """
        assert base_vjp is not None
        raw_grads = base_vjp(graph, node, cotangent)
        return tuple(
            _bcast_reduce_to_input(graph, str(adj_id), inp_id)
            for adj_id, inp_id in zip(raw_grads, node.inputs)
        )

    return _wrapped


def _reshape_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str | UnconnectedGradients, ...]:
    """Compute the VJP of `Reshape` by reshaping `cotangent` back to the input's shape.

    Args:
        graph: Active backward IR graph.
        node: Forward `Reshape` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple containing the reshaped gradient node ID and `ZERO` for any extra shape inputs.
    """
    inp_id = node.inputs[0]
    inp_shape = _extract_node_shape(graph, inp_id)
    extra_zeros = tuple(UnconnectedGradients.ZERO for _ in node.inputs[1:])
    reshaped = emit_ir_node(
        graph,
        "Reshape",
        [cotangent],
        shape_metadata=inp_shape,
        attributes={"newshape": list(inp_shape)},
    )
    return (reshaped, *extra_zeros)


def _expand_dims_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str]:
    """Compute the VJP of `ExpandDims` by squeezing the inserted dimension.

    Args:
        graph: Active backward IR graph.
        node: Forward `ExpandDims` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Single-element tuple containing the squeezed gradient node ID.
    """
    inp_id = node.inputs[0]
    inp_shape = _extract_node_shape(graph, inp_id)
    axis = node.attributes.get("axis", 0)
    squeezed = emit_ir_node(
        graph,
        "Squeeze",
        [cotangent],
        shape_metadata=inp_shape,
        attributes={"axis": axis},
    )
    return (squeezed,)


def _zero_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[UnconnectedGradients, ...]:
    """Return zero gradients for non-differentiable creation, comparison, and piecewise step ops.

    Args:
        graph: Active backward IR graph.
        node: Non-differentiable forward node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple of `UnconnectedGradients.ZERO` matching `len(node.inputs)`.
    """
    del graph, cotangent
    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


def _cast_vjp_rule(
    graph: LogicalGraph,
    node: LogicalNode,
    cotangent: str,
) -> tuple[str | UnconnectedGradients, ...]:
    """Pass `cotangent` through `Cast` during reverse-mode differentiation.

    Args:
        graph: Active backward IR graph.
        node: Forward `Cast` node.
        cotangent: Incoming cotangent node ID.

    Returns:
        Tuple `(cotangent, *extra_zeros)`.
    """
    del graph
    return (cotangent, *(UnconnectedGradients.ZERO for _ in node.inputs[1:]))


def _register_autodiff_vjp_rules() -> None:
    """Register complete VJP rules in `ml_switcheroo_compiler` for `zero_jax.grad` support."""
    _VJP_REGISTRY["Sum"] = _sum_vjp_rule
    _VJP_REGISTRY["ReduceSum"] = _sum_vjp_rule
    _VJP_REGISTRY["Mean"] = _mean_vjp_rule
    _VJP_REGISTRY["ReduceMean"] = _mean_vjp_rule
    _VJP_REGISTRY["Max"] = _extrema_vjp_rule
    _VJP_REGISTRY["ReduceMax"] = _extrema_vjp_rule
    _VJP_REGISTRY["Min"] = _extrema_vjp_rule
    _VJP_REGISTRY["ReduceMin"] = _extrema_vjp_rule
    _VJP_REGISTRY["Where"] = _where_vjp_rule
    _VJP_REGISTRY["Maximum"] = _maximum_vjp_rule
    _VJP_REGISTRY["Minimum"] = _minimum_vjp_rule
    _VJP_REGISTRY["Power"] = _power_vjp_rule
    _VJP_REGISTRY["Pow"] = _power_vjp_rule
    _VJP_REGISTRY["Reshape"] = _reshape_vjp_rule
    _VJP_REGISTRY["ExpandDims"] = _expand_dims_vjp_rule
    _VJP_REGISTRY["Cast"] = _cast_vjp_rule
    for bin_op in (
        "Add",
        "Subtract",
        "Sub",
        "Multiply",
        "Mul",
        "Divide",
        "TrueDivide",
        "Div",
    ):
        _VJP_REGISTRY[bin_op] = _make_bcast_binary_vjp(bin_op)
    for non_diff_op in (
        "Zeros",
        "ZerosLike",
        "Ones",
        "OnesLike",
        "Full",
        "FullLike",
        "Arange",
        "Eye",
        "Equal",
        "NotEqual",
        "Greater",
        "GreaterEqual",
        "Less",
        "LessEqual",
        "LogicalAnd",
        "LogicalOr",
        "LogicalNot",
        "IsNaN",
        "IsInf",
        "IsFinite",
        "Floor",
        "Ceil",
        "Round",
        "Sign",
    ):
        _VJP_REGISTRY[non_diff_op] = _zero_vjp_rule


_register_autodiff_vjp_rules()


def _resolve_dtype(dtype: DTypeLike) -> DType | None:
    """Resolve a `DTypeLike` value into an `ml_switcheroo_ir` `DType` enum or None.

    Args:
        dtype: Optional `DType`, string name, or Python numeric type.

    Returns:
        Resolved `DType` instance or None.
    """
    if dtype is None or isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, DTypeLikeProtocol):
        return DType(dtype.name)
    if isinstance(dtype, str):
        clean = dtype[6:] if dtype.startswith("DType.") else dtype
        return DType(clean)
    if dtype is bool:
        return DType("bool")
    if dtype is float:
        return DType.float32
    return DType.int32


def _extract_array_dtype(
    val: ArrayLikeProtocol | ToListArrayProtocol,
    override_dtype: DTypeLike,
) -> DType | None:
    """Resolve the target `DType` for an `ArrayLikeProtocol` or `ToListArrayProtocol` value.

    Args:
        val: Array-like object with a `.dtype` property.
        override_dtype: Optional explicit override dtype.

    Returns:
        Resolved `DType` or None.
    """
    if override_dtype is not None:
        return _resolve_dtype(override_dtype)
    raw_dt = val.dtype
    if isinstance(raw_dt, DType):
        return raw_dt
    return _resolve_dtype(str(raw_dt) if raw_dt is not None else None)


RawNestedScalar = Union[Tensor, int, float, bool, Sequence["RawNestedScalar"]]


def _to_raw_array_input(
    val: ArrayLike | JaxArrayLikeSpec | OrbaxPyTree,
) -> RawNestedScalar:
    """Recursively unwrap an `ArrayLike` into a `Tensor`, scalar, or nested sequence of scalars.

    Args:
        val: Input `ArrayLike` value.

    Returns:
        Unwrapped `Tensor`, scalar, or nested list of scalars.

    Raises:
        TypeError: If `val` is not a supported `ArrayLike` value.
    """
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        return [_to_raw_array_input(item) for item in val]
    raise TypeError(f"Unsupported ArrayLike input type: {type(val)!r}.")


def asarray(val: ArrayLike, dtype: DTypeLike = None) -> Tensor:
    """Convert an `ArrayLike` value into an `ml-switcheroo-compiler` `Tensor`.

    Preserves `zero_jax.numpy.ndarray` (`JaxArray`) and traced `ProxyTensor` graphs
    directly without `.tolist()` host copies so `zero_jax.grad` and `zero_jax.jit`
    trace seamlessly through all `zero-optax` losses and transformations.

    Args:
        val: Input tensor, scalar, variable leaf, `zero_jax` array, or nested sequence.
        dtype: Optional target data type.

    Returns:
        Converted `Tensor` instance.
    """
    if isinstance(val, VariableLeafProtocol):
        val = val.value
    target_dtype = _resolve_dtype(dtype)
    if isinstance(val, JaxArray):
        raw_tensor = _jax_to_tensor(val)
        if target_dtype is None or raw_tensor.dtype == target_dtype:
            return raw_tensor
        return _jax_to_tensor(lax.convert_element_type(val, target_dtype.value))
    if isinstance(val, Tensor):
        lifted = _jax_to_tensor(val)
        if target_dtype is None or lifted.dtype == target_dtype:
            return lifted
        return _jax_to_tensor(
            lax.convert_element_type(JaxArray(lifted), target_dtype.value)
        )
    if (
        isinstance(val, (list, tuple))
        and len(val) > 0
        and any(
            isinstance(item, (JaxArray, Tensor, VariableLeafProtocol)) for item in val
        )
    ):
        return stack([asarray(item, dtype=target_dtype) for item in val], axis=0)
    if isinstance(val, ArrayLikeProtocol):
        return as_tensor_obj(
            ops.asarray(
                _to_raw_array_input(val.tolist()),
                dtype=_extract_array_dtype(val, dtype),
            )
        )
    if isinstance(val, ToListArrayProtocol):
        flat_list: list[int | float | bool] = [
            float(x.real) if isinstance(x, complex) else x for x in val.to_list()
        ]
        flat_tensor = as_tensor_obj(
            ops.asarray(flat_list, dtype=_extract_array_dtype(val, dtype))
        )
        return as_tensor_obj(
            shape_reshape.Reshape()(flat_tensor, newshape=tuple(val.shape))
        )
    return as_tensor_obj(ops.asarray(_to_raw_array_input(val), dtype=target_dtype))


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
        return as_tensor_obj(ops.arange(0, start, step, dtype=DType.int32))
    return as_tensor_obj(ops.arange(start, stop, step, dtype=DType.int32))


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


def clip(
    x: ArrayLike,
    a_min: ArrayLike | None = None,
    a_max: ArrayLike | None = None,
) -> Tensor:
    """Clip (limit) the values in an array between `a_min` and `a_max`.

    Args:
        x: Input tensor or scalar.
        a_min: Optional minimum value bound.
        a_max: Optional maximum value bound.

    Returns:
        Clipped Tensor.
    """
    out = asarray(x)
    if a_min is not None:
        out = maximum(out, a_min)
    if a_max is not None:
        out = minimum(out, a_max)
    return out


def abs_val(x: ArrayLike) -> Tensor:
    """Compute element-wise absolute value.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise absolute value Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.abs(x))
    return as_tensor_obj(ops.abs(asarray(x)))


abs_tensor = abs_val


def exp(x: ArrayLike) -> Tensor:
    """Compute element-wise exponential.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise exponential Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.exp(x))
    return as_tensor_obj(ops.exp(asarray(x)))


def log(x: ArrayLike) -> Tensor:
    """Compute element-wise natural logarithm.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise natural logarithm Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.log(x))
    return as_tensor_obj(ops.log(asarray(x)))


def log1p(x: ArrayLike) -> Tensor:
    """Compute element-wise log(1 + x) accurately for small x.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise log(1 + x) Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.log1p(x))
    return as_tensor_obj(ops.log1p(asarray(x)))


def sqrt(x: ArrayLike) -> Tensor:
    """Compute element-wise non-negative square root.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise square root Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.sqrt(x))
    return as_tensor_obj(ops.sqrt(asarray(x)))


def sign(x: ArrayLike) -> Tensor:
    """Compute element-wise sign indication (-1, 0, +1).

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise sign Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.sign(x))
    return as_tensor_obj(ops.sign(asarray(x)))


def cos(x: ArrayLike) -> Tensor:
    """Compute element-wise cosine.

    Args:
        x: Input tensor or scalar in radians.

    Returns:
        Element-wise cosine Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.cos(x))
    return as_tensor_obj(ops.cos(asarray(x)))


def sin(x: ArrayLike) -> Tensor:
    """Compute element-wise sine.

    Args:
        x: Input tensor or scalar in radians.

    Returns:
        Element-wise sine Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.sin(x))
    return as_tensor_obj(ops.sin(asarray(x)))


def floor(x: ArrayLike) -> Tensor:
    """Compute element-wise largest integer less than or equal to `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Element-wise floor Tensor.
    """
    if isinstance(x, JaxArray):
        return _jax_to_tensor(lax.floor(x))
    return as_tensor_obj(ops.floor(asarray(x)))


def power(x: ArrayLike, exponent: ArrayLike) -> Tensor:
    """Raise elements of `x` to powers from `exponent` element-wise.

    Args:
        x: Base tensor or scalar.
        exponent: Exponent tensor or scalar.

    Returns:
        Element-wise `x ** exponent` Tensor.
    """
    return as_tensor_obj(ops.power(asarray(x), asarray(exponent)))


def concatenate(tensors: Sequence[ArrayLike], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along an existing axis.

    Args:
        tensors: Non-empty sequence of tensors with matching shapes except in `axis`.
        axis: The axis along which the tensors will be joined.

    Returns:
        Concatenated Tensor.
    """
    return as_tensor_obj(ops.concatenate([asarray(t) for t in tensors], axis=axis))


def stack(tensors: Sequence[ArrayLike], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along a new axis.

    Args:
        tensors: Non-empty sequence of tensors with identical shapes.
        axis: The axis in the result tensor along which the input tensors are stacked.

    Returns:
        Stacked Tensor.
    """
    t_list = [asarray(t) for t in tensors]
    if any(isinstance(t.data, ProxyTensor) for t in t_list) or _tracer.is_tracing:
        expanded = [expand_dims(t, axis) for t in t_list]
        n_items = len(expanded)
        out_ndim = expanded[0].ndim
        norm_ax = axis if axis >= 0 else out_ndim + axis
        idx_shape = [1] * out_ndim
        idx_shape[norm_ax] = n_items
        k_idx = reshape(arange(n_items), idx_shape)
        acc = where(k_idx == 0, expanded[0], 0.0)
        for idx_i in range(1, n_items):
            acc = acc + where(k_idx == idx_i, expanded[idx_i], 0.0)
        return acc
    return as_tensor_obj(ops.stack(t_list, axis=axis))


def sigmoid(x: ArrayLike) -> Tensor:
    """Compute numerically stable element-wise logistic sigmoid.

    Args:
        x: Input tensor or scalar.

    Returns:
        Sigmoid probability Tensor in [0, 1].
    """
    t = asarray(x)
    neg_abs = -abs_val(t)
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
    t = asarray(x)
    if isinstance(axis, tuple) and (
        isinstance(t.data, ProxyTensor) or _tracer.is_tracing
    ):
        cur = t
        norm_axes = sorted(
            (a if a >= 0 else t.ndim + a for a in axis),
            reverse=not keepdims,
        )
        for ax in norm_axes:
            cur = as_tensor_obj(ops.sum(cur, axis=ax, keepdims=keepdims))
        return cur
    return as_tensor_obj(ops.sum(t, axis=axis, keepdims=keepdims))


def sum_to_Float(x: ArrayLike) -> Tensor:
    """Compute the scalar floating-point sum of all elements in `x`.

    Args:
        x: Input tensor or array-like.

    Returns:
        Scalar floating-point sum Tensor.
    """
    return reduce_sum(asarray(x, dtype="float32"))


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


mean_axis = reduce_mean


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


def reduce_min(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute minimum of tensor elements over a given axis or axes.

    Args:
        x: Input tensor or array-like.
        axis: Axis or tuple of axes along which the minimum is computed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced minimum Tensor.
    """
    return as_tensor_obj(ops.min(asarray(x), axis=axis, keepdims=keepdims))


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
    t = asarray(x)
    if (isinstance(t.data, ProxyTensor) or _tracer.is_tracing) and axis is not None:
        return squeeze_axes(t, axis)
    return as_tensor_obj(ops.squeeze(t, axis=axis))


def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis: int) -> Tensor:
    """Take values from the input tensor by matching 1D index and data slices.

    Uses differentiable mask-gather operations when `axis` is the last dimension
    during compiler tracing so `zero_jax.grad` differentiates through indexed losses.

    Args:
        arr: Source tensor.
        indices: Indices to take along each 1D slice of `arr`.
        axis: The axis to take 1D slices along.

    Returns:
        Indexed result wrapped as a Tensor.
    """
    arr_t = asarray(arr)
    idx_t = asarray(indices)
    ndim = arr_t.ndim
    norm_ax = axis if axis >= 0 else ndim + axis
    if (
        (isinstance(arr_t.data, ProxyTensor) or _tracer.is_tracing)
        and ndim >= 1
        and norm_ax == ndim - 1
        and 0 not in arr_t.shape
    ):
        dim_k = arr_t.shape[-1]
        k_range = reshape(arange(dim_k), (1,) * ndim + (dim_k,))
        idx_exp = expand_dims(idx_t, -1)
        one_hot = idx_exp == k_range
        arr_exp = expand_dims(arr_t, -2)
        return reduce_sum(where(one_hot, arr_exp, 0.0), axis=-1)
    raw = indexing.take_along_axis(arr_t, idx_t, axis=axis)
    return asarray(as_tensor_obj(raw))


def reshape(x: ArrayLike, newshape: Sequence[int]) -> Tensor:
    """Give a new shape to a tensor without changing its data.

    Args:
        x: Input tensor to be reshaped.
        newshape: New shape tuple or sequence of ints.

    Returns:
        Reshaped Tensor.
    """
    return as_tensor_obj(shape_reshape.Reshape()(asarray(x), newshape=tuple(newshape)))


LeafFn1 = Callable[[TensorLeaf], TensorLeaf]
LeafFn2 = Callable[[TensorLeaf, TensorLeaf], TensorLeaf]
LeafFn3 = Callable[[TensorLeaf, TensorLeaf, TensorLeaf], TensorLeaf]
LeafFn4 = Callable[[TensorLeaf, TensorLeaf, TensorLeaf, TensorLeaf], TensorLeaf]
LeafFn = Union[LeafFn1, LeafFn2, LeafFn3, LeafFn4]


def _is_leaf_fn1(fn: LeafFn, n_extra: int) -> TypeGuard[LeafFn1]:
    """Narrow `fn` to `LeafFn1` when `n_extra == 0`.

    Args:
        fn: Leaf transformation callable.
        n_extra: Number of extra leaf arguments beyond the first.

    Returns:
        True if `n_extra == 0`.
    """
    del fn
    return n_extra == 0


def _is_leaf_fn2(fn: LeafFn, n_extra: int) -> TypeGuard[LeafFn2]:
    """Narrow `fn` to `LeafFn2` when `n_extra == 1`.

    Args:
        fn: Leaf transformation callable.
        n_extra: Number of extra leaf arguments beyond the first.

    Returns:
        True if `n_extra == 1`.
    """
    del fn
    return n_extra == 1


def _is_leaf_fn3(fn: LeafFn, n_extra: int) -> TypeGuard[LeafFn3]:
    """Narrow `fn` to `LeafFn3` when `n_extra == 2`.

    Args:
        fn: Leaf transformation callable.
        n_extra: Number of extra leaf arguments beyond the first.

    Returns:
        True if `n_extra == 2`.
    """
    del fn
    return n_extra == 2


def _is_leaf_fn4(fn: LeafFn, n_extra: int) -> TypeGuard[LeafFn4]:
    """Narrow `fn` to `LeafFn4` when `n_extra == 3`.

    Args:
        fn: Leaf transformation callable.
        n_extra: Number of extra leaf arguments beyond the first.

    Returns:
        True if `n_extra == 3`.
    """
    del fn
    return n_extra == 3


def _unwrap_leaf(leaf: TensorLeaf) -> TensorLeaf:
    """Unwrap a `VariableLeafProtocol` leaf to its inner `Tensor` value if applicable.

    Args:
        leaf: Input leaf value.

    Returns:
        Unwrapped `Tensor` if `leaf` is a `VariableLeafProtocol`, otherwise `leaf`.
    """
    if isinstance(leaf, VariableLeafProtocol):
        return asarray(leaf.value)
    return leaf


def _replace_variable_leaf(
    leaf: VariableLeafProtocol,
    new_val: PyTree,
) -> VariableLeafProtocol:
    """Create a new `VariableLeafProtocol` container of the same type holding `new_val`.

    Uses `leaf.replace(...)` on `zero_flax.nnx.Variable` instances to preserve variable
    metadata (`tag`, `role`, `lora_rank`, etc.) and wraps `Tensor` values in
    `zero_jax.numpy.ndarray` (`JaxArray`) when the original variable leaf held a `JaxArray`.

    Args:
        leaf: Original variable leaf container (`Param`, `BatchStat`, `LoRAParam`).
        new_val: Replacement tensor or leaf value.

    Returns:
        New variable container of the same class wrapping `new_val`.
    """
    wrapped_val: PyTree = (
        JaxArray(new_val)
        if isinstance(new_val, Tensor) and isinstance(leaf.value, JaxArray)
        else new_val
    )
    if isinstance(leaf, Variable) and isinstance(
        wrapped_val, (JaxArray, Tensor, int, float, bool, type(None))
    ):
        replaced_leaf: VariableLeafProtocol = leaf.replace(wrapped_val)
        return replaced_leaf
    cls_type = type(leaf)
    return cls_type(wrapped_val)


def _coerce_pytree_child(child: CustomMappingChild | MaskTree | ArrayLike) -> PyTree:
    """Narrow a child extracted from a `CustomMappingPyTreeProtocol` or `MappableDataclassProtocol` into a `PyTree`.

    Args:
        child: Child node extracted from a custom mapping or mappable dataclass.

    Returns:
        Validated `PyTree` node.
    """
    if isinstance(
        child,
        (
            Tensor,
            JaxArray,
            float,
            int,
            bool,
            type(None),
            MaskedNode,
            ArrayLikeProtocol,
            ToListArrayProtocol,
            ShapeDtypeProtocol,
            VariableLeafProtocol,
            CustomMappingPyTreeProtocol,
            MappableDataclassProtocol,
        ),
    ) and not isinstance(child, dict):
        return child
    if isinstance(child, Mapping):
        return {str(k): _coerce_pytree_child(v) for k, v in child.items()}
    if isinstance(child, tuple):
        return tuple(_coerce_pytree_child(item) for item in child)
    if isinstance(child, list):
        return [_coerce_pytree_child(item) for item in child]
    return None


def _apply_leaf_fn(
    fn: LeafFn,
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

    Raises:
        ValueError: If `len(rest_leaves) > 3`.
    """
    u_first = _unwrap_leaf(first)
    u_rest = [_unwrap_leaf(r) for r in rest_leaves]
    if isinstance(u_first, MaskedNode) or any(
        isinstance(r, MaskedNode) for r in u_rest
    ):
        return MaskedNode()
    n_extra = len(u_rest)
    if _is_leaf_fn1(fn, n_extra):
        res = fn(u_first)
    elif _is_leaf_fn2(fn, n_extra):
        res = fn(u_first, u_rest[0])
    elif _is_leaf_fn3(fn, n_extra):
        res = fn(u_first, u_rest[0], u_rest[1])
    elif _is_leaf_fn4(fn, n_extra):
        res = fn(u_first, u_rest[0], u_rest[1], u_rest[2])
    else:
        raise ValueError(
            f"tree_map supports between 1 and 4 PyTrees, got {1 + n_extra}."
        )
    if (
        isinstance(first, VariableLeafProtocol)
        and res is not None
        and not isinstance(res, (VariableLeafProtocol, MaskedNode))
    ):
        return _replace_variable_leaf(first, res)
    return res


def _rebuild_mapping(
    template: Mapping[str, PyTree] | CustomMappingPyTreeProtocol,
    result_dict: dict[str, PyTree],
) -> PyTree:
    """Reconstruct a mapping container matching `template` (`dict`, `State`, or `FrozenDict`).

    Args:
        template: Original mapping or custom mapping container.
        result_dict: Mapped key-value dictionary.

    Returns:
        Reconstructed mapping container of the same type as `template`.
    """
    if (
        isinstance(template, CustomMappingPyTreeProtocol)
        and type(template).__name__ != "dict"
    ):
        cls_type = type(template)
        return cls_type(result_dict)
    return result_dict


def _rebuild_sequence(
    template: tuple[PyTree, ...] | list[PyTree],
    items: list[PyTree],
) -> PyTree:
    """Reconstruct a list, tuple, or NamedTuple matching `template`'s container type.

    Args:
        template: Original sequence container (`list`, `tuple`, or `NamedTuple`).
        items: Mapped child elements.

    Returns:
        Reconstructed sequence or `NamedTuple` instance.
    """
    if isinstance(template, list):
        return list(items)
    if isinstance(template, NamedTupleProtocol):
        return template._make(items)
    return tuple(items)


def _extract_rest_seqs(
    tree_len: int,
    rest: Sequence[PyTree],
) -> list[tuple[PyTree, ...] | list[PyTree]]:
    """Validate that all `rest` items are sequences of length `tree_len`.

    Args:
        tree_len: Expected sequence length.
        rest: Co-traversed PyTree sequences.

    Returns:
        List of validated sequence containers from `rest`.

    Raises:
        ValueError: If any element in `rest` is not a tuple/list of length `tree_len`.
    """
    rest_seqs: list[tuple[PyTree, ...] | list[PyTree]] = []
    for r in rest:
        if not isinstance(r, (tuple, list)) or len(r) != tree_len:
            raise ValueError(
                "PyTree structure mismatch: expected sequence of matching length."
            )
        rest_seqs.append(r)
    return rest_seqs


def _extract_leaf(node: PyTree) -> TensorLeaf:
    """Validate that `node` is a leaf and return it as a `TensorLeaf`.

    Args:
        node: Co-traversed PyTree node expected to be a leaf.

    Returns:
        The leaf value.

    Raises:
        ValueError: If `node` is a Mapping, custom mapping, dataclass, tuple, or list container.
    """
    if not isinstance(node, MaskedNode) and isinstance(
        node,
        (
            MappableDataclassProtocol,
            CustomMappingPyTreeProtocol,
            Mapping,
            tuple,
            list,
        ),
    ):
        raise ValueError(
            "PyTree structure mismatch: expected leaf node across all trees."
        )
    return node


def _is_optax_leaf(node: JaxArrayLikeSpec) -> bool:
    """Check whether `node` should be treated as an atomic PyTree leaf in `zero_jax.tree_util`.

    Args:
        node: Candidate PyTree node.

    Returns:
        True if `node` is `None`, `MaskedNode`, or `VariableLeafProtocol`.
    """
    return node is None or isinstance(node, (MaskedNode, VariableLeafProtocol))


def tree_map(
    fn: LeafFn,
    tree: PyTree,
    *rest: PyTree,
) -> PyTree:
    """Apply a function to every leaf across one or more PyTrees with matching structure.

    Delegates structural flattening to `zero_jax.tree_util` while preserving
    `MaskedNode`, `zero_flax.nnx.Variable`, `zero_flax.nnx.State`, and `FrozenDict`
    container and leaf semantics.

    Args:
        fn: Callable taking 1 to 4 leaves (one per input tree) and returning a transformed leaf.
        tree: Primary PyTree structure.
        *rest: Additional PyTrees with identical container structure (up to 3 extra trees).

    Returns:
        New PyTree with the same structure as `tree` and transformed leaves.

    Raises:
        ValueError: If any PyTree in `rest` has a structure or length mismatch with `tree`,
            or if more than 4 PyTrees are passed.
    """
    return _tree_map_impl(fn, tree, rest)


def _tree_map_impl(
    fn: LeafFn,
    tree: PyTree,
    rest: Sequence[PyTree],
) -> PyTree:
    """Recursively traverse `tree` and `rest` and apply `fn` at the leaves.

    Args:
        fn: Leaf transformation callable taking `1 + len(rest)` arguments.
        tree: Primary PyTree node.
        rest: Sequence of co-traversed PyTree nodes.

    Returns:
        Transformed PyTree node matching `tree`.

    Raises:
        ValueError: If container structures or lengths do not match across `tree` and `rest`.
    """
    if isinstance(tree, MappableDataclassProtocol):
        field_names = tree.__chex_mappable_fields__()
        field_values = [_coerce_pytree_child(v) for v in tree.__chex_field_values__()]
        primary_fields = dict(zip(field_names, field_values))
        rest_field_maps: list[dict[str, PyTree]] = []
        for r in rest:
            if (
                isinstance(r, MappableDataclassProtocol)
                and r.__chex_mappable_fields__() == field_names
            ):
                r_vals = [_coerce_pytree_child(v) for v in r.__chex_field_values__()]
                rest_field_maps.append(dict(zip(r.__chex_mappable_fields__(), r_vals)))
            else:
                raise ValueError(
                    "PyTree structure mismatch: expected MappableDataclass with matching fields."
                )
        dc_result: dict[str, PyTree] = {}
        for k in field_names:
            other_vals = [rm[k] for rm in rest_field_maps]
            dc_result[k] = _tree_map_impl(fn, primary_fields[k], other_vals)
        dc_cls = type(tree)
        return dc_cls(**dc_result)
    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        result_dict: dict[str, PyTree] = {}
        tree_keys = set(tree.keys())
        rest_maps: list[Mapping[str, PyTree] | CustomMappingPyTreeProtocol] = []
        for r in rest:
            if (
                not isinstance(r, (Mapping, CustomMappingPyTreeProtocol))
                or set(r.keys()) != tree_keys
            ):
                raise ValueError(
                    "PyTree structure mismatch: expected Mapping with matching keys."
                )
            rest_maps.append(r)
        for k in tree:
            other_vals = [_coerce_pytree_child(rm[k]) for rm in rest_maps]
            result_dict[k] = _tree_map_impl(
                fn, _coerce_pytree_child(tree[k]), other_vals
            )
        return _rebuild_mapping(tree, result_dict)
    if isinstance(tree, MaskedNode):
        leaf_rest_m: list[TensorLeaf] = [_extract_leaf(r) for r in rest]
        return _apply_leaf_fn(fn, tree, leaf_rest_m)
    if isinstance(tree, (tuple, list)):
        seq_items: list[PyTree] = []
        rest_seqs = _extract_rest_seqs(len(tree), rest)
        for idx, item in enumerate(tree):
            other_items: list[PyTree] = [r_seq[idx] for r_seq in rest_seqs]
            seq_items.append(_tree_map_impl(fn, item, other_items))
        return _rebuild_sequence(tree, seq_items)
    leaf_rest: list[TensorLeaf] = [_extract_leaf(r) for r in rest]
    return _apply_leaf_fn(fn, tree, leaf_rest)


def tree_leaves(tree: PyTree) -> list[TensorLeaf]:
    """Collect all leaf values from a PyTree in depth-first order using `zero_jax.tree_util`.

    Args:
        tree: Input PyTree structure.

    Returns:
        Flat list of `TensorLeaf` values (excluding `MaskedNode`).
    """
    if isinstance(tree, MappableDataclassProtocol):
        dc_leaves: list[TensorLeaf] = []
        for fval in tree.__chex_field_values__():
            dc_leaves.extend(tree_leaves(_coerce_pytree_child(fval)))
        return dc_leaves
    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        map_leaves: list[TensorLeaf] = []
        for k in tree:
            map_leaves.extend(tree_leaves(_coerce_pytree_child(tree[k])))
        return map_leaves
    if isinstance(tree, (tuple, list)) and not isinstance(tree, MaskedNode):
        seq_leaves: list[TensorLeaf] = []
        for item in tree:
            seq_leaves.extend(tree_leaves(item))
        return seq_leaves
    if isinstance(tree, MaskedNode):
        return []
    jax_leaf: Tensor | None = None if tree is None else asarray(tree)
    raw_leaves = jax_tree_util.tree_leaves(jax_leaf, is_leaf=_is_optax_leaf)
    return [tree for _ in raw_leaves]


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
        if x is None or isinstance(x, MaskedNode):
            return x
        return zeros_like(x, dtype=dtype)

    return tree_map(_zero_leaf, tree)


def tree_ones_like(tree: Params, dtype: DTypeLike = None) -> Updates:
    """Create an identical PyTree structure where every leaf is a ones Tensor.

    Args:
        tree: Input parameter PyTree.
        dtype: Optional override data type for ones tensors.

    Returns:
        Ones-initialized PyTree matching `tree`.
    """

    def _one_leaf(x: TensorLeaf) -> TensorLeaf:
        """Create a ones Tensor matching leaf `x`.

        Args:
            x: Leaf value from parameter PyTree.

        Returns:
            Ones Tensor matching `x`.
        """
        if x is None or isinstance(x, MaskedNode):
            return x
        return ones_like(x, dtype=dtype)

    return tree_map(_one_leaf, tree)


def tree_add(first: PyTree, *rest: PyTree) -> PyTree:
    """Add two or more PyTrees element-wise across corresponding leaves.

    Args:
        first: First PyTree.
        *rest: Additional PyTrees to add to `first`.

    Returns:
        Element-wise sum PyTree.
    """
    acc = first
    for other in rest:
        acc = tree_map(
            lambda x, y: None if x is None or y is None else asarray(x) + asarray(y),
            acc,
            other,
        )
    return acc


def tree_sub(first: PyTree, second: PyTree) -> PyTree:
    """Subtract `second` from `first` element-wise across corresponding leaves.

    Args:
        first: Minuend PyTree.
        second: Subtrahend PyTree.

    Returns:
        Element-wise difference PyTree `first - second`.
    """
    return tree_map(
        lambda x, y: None if x is None or y is None else asarray(x) - asarray(y),
        first,
        second,
    )


def tree_mul(first: PyTree, second: PyTree) -> PyTree:
    """Multiply two PyTrees element-wise across corresponding leaves.

    Args:
        first: First PyTree.
        second: Second PyTree.

    Returns:
        Element-wise product PyTree `first * second`.
    """
    return tree_map(
        lambda x, y: None if x is None or y is None else asarray(x) * asarray(y),
        first,
        second,
    )


def tree_scale(scalar: Scalar | ArrayLike, tree: PyTree) -> PyTree:
    """Multiply every leaf in `tree` by `scalar`.

    Args:
        scalar: Scalar factor (Python float/int or scalar Tensor/JaxArray).
        tree: Input PyTree to scale.

    Returns:
        Scaled PyTree.
    """
    s_t = asarray(scalar)
    return tree_map(lambda x: None if x is None else s_t * asarray(x), tree)


def tree_vdot(first: PyTree, second: PyTree) -> Tensor:
    """Compute the inner (dot) product between two PyTrees of tensors.

    Args:
        first: First PyTree.
        second: Second PyTree with identical structure.

    Returns:
        Scalar Tensor holding the sum of element-wise products across all leaves.
    """
    prod_tree = tree_mul(first, second)
    total: Tensor = asarray(0.0)
    for leaf in tree_leaves(prod_tree):
        unwrapped = _unwrap_leaf(leaf)
        if unwrapped is not None and not isinstance(unwrapped, MaskedNode):
            total = total + reduce_sum(asarray(unwrapped))
    return total


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
        unwrapped = _unwrap_leaf(leaf)
        if unwrapped is not None and not isinstance(unwrapped, MaskedNode):
            t = asarray(unwrapped)
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
    unwrapped = _unwrap_leaf(x)
    if unwrapped is None or isinstance(unwrapped, MaskedNode):
        return asarray(0.0)
    t = asarray(unwrapped)
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


def _is_filter_spec(
    obj: MaskOrFn | PyTree,
) -> TypeGuard[Filter | type[VariableBase]]:
    """Check whether `obj` is a `zero_flax.nnx.Filter` instance or `VariableBase` subclass.

    Args:
        obj: Candidate mask specification.

    Returns:
        True if `obj` is a `Filter` or `VariableBase` subclass.
    """
    return isinstance(obj, Filter) or (
        isinstance(obj, type) and issubclass(obj, VariableBase)
    )


def evaluate_filter_mask_tree(
    filter_spec: Filter | type[VariableBase],
    params: Params,
    prefix: tuple[str, ...] = (),
) -> MaskTree:
    """Evaluate a `zero_flax.nnx.Filter` predicate across `params` to build a boolean `MaskTree`.

    Args:
        filter_spec: `Filter` instance or `VariableBase` subclass (e.g., `OfType(Param)`).
        params: Target parameters PyTree.
        prefix: Current tuple of string path segments from the root.

    Returns:
        Boolean `MaskTree` matching the container structure of `params`.
    """
    predicate = to_predicate(filter_spec)
    if isinstance(params, MappableDataclassProtocol):
        return tuple(
            evaluate_filter_mask_tree(
                predicate,
                _coerce_pytree_child(val),
                (*prefix, fname),
            )
            for fname, val in zip(
                params.__chex_mappable_fields__(),
                params.__chex_field_values__(),
            )
        )
    if isinstance(params, (Mapping, CustomMappingPyTreeProtocol)):
        return {
            str(k): evaluate_filter_mask_tree(
                predicate,
                _coerce_pytree_child(params[k]),
                (*prefix, str(k)),
            )
            for k in params
        }
    if isinstance(params, tuple):
        return tuple(
            evaluate_filter_mask_tree(predicate, item, (*prefix, str(idx)))
            for idx, item in enumerate(params)
        )
    if isinstance(params, list):
        return [
            evaluate_filter_mask_tree(predicate, item, (*prefix, str(idx)))
            for idx, item in enumerate(params)
        ]
    if isinstance(params, (VariableBase, JaxArray, int, float, bool, type(None))):
        return bool(predicate(prefix, params))
    return bool(predicate(prefix, None))


def resolve_mask(mask: MaskOrFn, params: Params) -> MaskTree | None:
    """Resolve an optional mask, `zero_flax.nnx.Filter`, or mask callable against `params`.

    Args:
        mask: Boolean mask PyTree, `zero_flax.nnx.Filter` / `Variable` type,
            callable `(Params) -> MaskTree`, or None.
        params: Current model parameters PyTree.

    Returns:
        Resolved `MaskTree` or None.
    """
    if mask is None:
        return None
    if isinstance(mask, (Filter, type)):
        return evaluate_filter_mask_tree(mask, params)
    if callable(mask):
        resolved = mask(params)
        if isinstance(resolved, (Filter, type)):
            return evaluate_filter_mask_tree(resolved, params)
        return resolved
    return mask


def _coerce_submask(val: PyTree | MaskTree) -> MaskTree:
    """Coerce a child mask node extracted from a custom mapping into a `MaskTree`.

    Args:
        val: Extracted child mask node.

    Returns:
        Validated `MaskTree`.
    """
    if isinstance(val, (bool, Filter, VariableLeafProtocol)):
        return val
    if isinstance(val, type) and issubclass(val, VariableBase):
        return val
    if isinstance(val, tuple):
        return tuple(_coerce_submask(item) for item in val)
    if isinstance(val, list):
        return [_coerce_submask(item) for item in val]
    if isinstance(val, (Mapping, CustomMappingPyTreeProtocol)):
        return {str(k): _coerce_submask(_coerce_pytree_child(val[k])) for k in val}
    return bool(val)


def apply_mask_tree(
    masked_tree: Updates,
    unmasked_tree: Updates,
    mask_tree: MaskTree,
) -> Updates:
    """Select leaves from `masked_tree` where `mask_tree` is True and `unmasked_tree` where False.

    Args:
        masked_tree: PyTree of updates with transformation applied.
        unmasked_tree: PyTree of fallback updates when masked out.
        mask_tree: Boolean, `Filter`, or nested structure of booleans matching the PyTree.

    Returns:
        Combined PyTree respecting the mask.
    """
    if _is_filter_spec(mask_tree):
        resolved_filter = evaluate_filter_mask_tree(mask_tree, unmasked_tree)
        return apply_mask_tree(masked_tree, unmasked_tree, resolved_filter)
    if isinstance(mask_tree, VariableLeafProtocol):
        return masked_tree if bool(mask_tree.value) else unmasked_tree
    if isinstance(mask_tree, bool):
        return masked_tree if mask_tree else unmasked_tree
    if (
        isinstance(mask_tree, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(masked_tree, (Mapping, CustomMappingPyTreeProtocol))
        and isinstance(unmasked_tree, (Mapping, CustomMappingPyTreeProtocol))
    ):
        mask_keys = set(mask_tree.keys())
        return _rebuild_mapping(
            masked_tree,
            {
                k: apply_mask_tree(
                    _coerce_pytree_child(masked_tree[k]),
                    _coerce_pytree_child(unmasked_tree[k]),
                    _coerce_submask(_coerce_pytree_child(mask_tree[k]))
                    if k in mask_keys
                    else True,
                )
                for k in masked_tree
            },
        )
    if (
        isinstance(mask_tree, tuple)
        and isinstance(masked_tree, tuple)
        and isinstance(unmasked_tree, tuple)
    ):
        return _rebuild_sequence(
            masked_tree,
            [
                apply_mask_tree(m_val, u_val, b_val)
                for m_val, u_val, b_val in zip(masked_tree, unmasked_tree, mask_tree)
            ],
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
    raise ValueError(
        "Mask tree container structure does not match masked_tree and unmasked_tree."
    )


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
    if isinstance(t.data, ProxyTensor) or _tracer.is_tracing:
        sum_exp_traced = reduce_sum(exp(t), axis=axis, keepdims=keepdims)
        return log(maximum(sum_exp_traced, 1e-30))
    c = reduce_max(t, axis=axis, keepdims=True)
    c_safe = where(abs_val(c) == INF, 0.0, c)
    shifted = where(abs_val(t) == INF, NEG_INF, t - c_safe)
    sum_exp = reduce_sum(exp(shifted), axis=axis, keepdims=keepdims)
    c_out = c_safe if keepdims else squeeze_axes(c_safe, axis)
    return c_out + log(where(sum_exp == 0.0, 1.0, sum_exp))


def log_softmax(
    logits: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where_mask: ArrayLike | None = None,
) -> Tensor:
    """Compute numerically stable log-softmax along `axis` handling `-inf` and optional mask.

    Args:
        logits: Unnormalized log probabilities.
        axis: Axis or axes along which the log-softmax is computed.
        where_mask: Optional boolean mask indicating valid elements to include.

    Returns:
        Log-probabilities Tensor matching the shape of `logits`.
    """
    l_arr = asarray(logits)
    if (
        isinstance(l_arr.data, ProxyTensor) or _tracer.is_tracing
    ) and where_mask is None:
        sum_exp_traced = reduce_sum(exp(l_arr), axis=axis, keepdims=True)
        return l_arr - log(maximum(sum_exp_traced, 1e-30))
    if where_mask is not None:
        w_arr = asarray(where_mask)
        l_arr = where(w_arr, l_arr, NEG_INF)
    c = reduce_max(l_arr, axis=axis, keepdims=True)
    c_safe = where(abs_val(c) == INF, 0.0, c)
    l_minus_c = where(abs_val(l_arr) == INF, NEG_INF, l_arr - c_safe)
    exp_l_c = exp(l_minus_c)
    if where_mask is not None:
        exp_l_c = where(asarray(where_mask), exp_l_c, 0.0)
    sumexp = reduce_sum(exp_l_c, axis=axis, keepdims=True)
    logsumexp_val = log(where(sumexp == 0.0, 1.0, sumexp))
    out = where(
        abs_val(l_arr) == INF,
        NEG_INF,
        l_minus_c - logsumexp_val,
    )
    if where_mask is not None:
        return where(asarray(where_mask), out, 0.0)
    return out


def softmax(
    logits: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where_mask: ArrayLike | None = None,
) -> Tensor:
    """Compute numerically stable softmax probabilities along `axis` with optional mask.

    Args:
        logits: Unnormalized log probabilities.
        axis: Axis or axes along which the softmax is computed.
        where_mask: Optional boolean mask indicating valid elements to include.

    Returns:
        Normalized probability distribution Tensor along `axis`.
    """
    probs = exp(log_softmax(logits, axis=axis, where_mask=where_mask))
    if where_mask is not None:
        return where(asarray(where_mask), probs, 0.0)
    return probs


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
    "abs_val",
    "apply_mask_tree",
    "arange",
    "asarray",
    "clip",
    "concatenate",
    "cos",
    "evaluate_filter_mask_tree",
    "evaluate_schedule",
    "exp",
    "expand_dims",
    "floor",
    "full_like",
    "leaf_l2_norm",
    "log",
    "log1p",
    "log_softmax",
    "logsumexp",
    "maximum",
    "mean_axis",
    "minimum",
    "ones",
    "ones_like",
    "power",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "reshape",
    "resolve_mask",
    "safe_norm",
    "sigmoid",
    "sign",
    "sin",
    "softmax",
    "sqrt",
    "squeeze",
    "squeeze_axes",
    "stack",
    "sum_to_Float",
    "take_along_axis",
    "tree_add",
    "tree_l2_norm",
    "tree_leaves",
    "tree_map",
    "tree_mul",
    "tree_ones_like",
    "tree_scale",
    "tree_sub",
    "tree_vdot",
    "tree_zeros_like",
    "where",
    "zeros",
    "zeros_like",
]
