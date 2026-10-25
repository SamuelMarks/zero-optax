"""Euclidean and constrained projection operators for `zero_optax.projections`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typing_extensions import TypeGuard

from zero_optax._tree import (
    abs_tensor,
    asarray,
    maximum,
    minimum,
    reshape,
    sign,
    tree_leaves,
    tree_map,
)
from zero_optax._types import (
    ArrayLike,
    CustomMappingPyTreeProtocol,
    MappableDataclassProtocol,
    PyTree,
    Scalar,
    TensorLeaf,
)
from zero_optax.tree_utils._random import _unflatten_like
from zero_optax.tree_utils._tree_math import (
    _flatten_tensor_floats,
    tree_add_scale,
    tree_full_like,
    tree_l1_norm,
    tree_l2_norm,
    tree_mul,
    tree_scalar_mul,
    tree_vdot,
)


def _is_scalar_bound(bound: PyTree) -> TypeGuard[ArrayLike]:
    """Check whether `bound` is a scalar or single-tensor bound rather than a container PyTree.

    Args:
        bound: Lower or upper bound argument.

    Returns:
        True if `bound` is a non-None leaf (`ArrayLike`) rather than a container PyTree.
    """
    return bound is not None and not isinstance(
        bound,
        (
            Mapping,
            CustomMappingPyTreeProtocol,
            MappableDataclassProtocol,
            tuple,
            list,
        ),
    )


def projection_non_negative(tree: PyTree) -> PyTree:
    """Project `tree` onto the non-negative orthant (`max(x, 0)`).

    Args:
        tree: Input PyTree to project.

    Returns:
        Projected PyTree with non-negative leaves.
    """

    def _relu_leaf(x: TensorLeaf) -> TensorLeaf:
        """Clamp leaf `x` below at 0.0.

        Args:
            x: Input leaf.

        Returns:
            Non-negative Tensor leaf.
        """
        return None if x is None else maximum(asarray(x), 0.0)

    return tree_map(_relu_leaf, tree)


def projection_box(
    tree: PyTree,
    lower: PyTree,
    upper: PyTree,
) -> PyTree:
    """Project `tree` onto the box constraint `[lower, upper]`.

    Args:
        tree: Input PyTree to project.
        lower: Scalar or PyTree lower bound.
        upper: Scalar or PyTree upper bound.

    Returns:
        Projected PyTree with elements clamped to `[lower, upper]`.
    """
    lower_tree: PyTree = (
        tree_full_like(tree, float(asarray(lower).item()))
        if _is_scalar_bound(lower)
        else (tree_map(lambda _: None, tree) if lower is None else lower)
    )
    upper_tree: PyTree = (
        tree_full_like(tree, float(asarray(upper).item()))
        if _is_scalar_bound(upper)
        else (tree_map(lambda _: None, tree) if upper is None else upper)
    )

    def _clip_box(x: TensorLeaf, lo: TensorLeaf, hi: TensorLeaf) -> TensorLeaf:
        """Clip leaf `x` to `[lo, hi]`.

        Args:
            x: Input leaf.
            lo: Lower bound leaf.
            hi: Upper bound leaf.

        Returns:
            Clipped Tensor leaf.
        """
        if x is None:
            return None
        res = asarray(x)
        if lo is not None:
            res = maximum(res, lo)
        if hi is not None:
            res = minimum(res, hi)
        return res

    return tree_map(_clip_box, tree, lower_tree, upper_tree)


def projection_hypercube(
    tree: PyTree,
    scale: PyTree = 1,
) -> PyTree:
    """Project `tree` onto the hypercube `[0, scale]^d`.

    Args:
        tree: Input PyTree to project.
        scale: Scalar or PyTree upper bound for the hypercube (default `1`).

    Returns:
        Projected PyTree with elements clamped to `[0, scale]`.
    """
    return projection_box(tree, lower=0.0, upper=scale)


def projection_linf_ball(
    tree: PyTree,
    scale: float = 1,
) -> PyTree:
    """Project `tree` onto the L-infinity ball `{y : ||y||_inf <= scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Radius of the L-infinity ball (default `1`).

    Returns:
        Projected PyTree with elements clamped to `[-scale, scale]`.
    """
    lower_tree = tree_full_like(tree, -float(scale))
    upper_tree = tree_full_like(tree, float(scale))
    return projection_box(tree, lower=lower_tree, upper=upper_tree)


def projection_l2_sphere(
    tree: PyTree,
    scale: float = 1,
) -> PyTree:
    """Project `tree` onto the L2 sphere `{y : ||y||_2 = scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Radius of the L2 sphere (default `1`).

    Returns:
        Projected PyTree normalized to L2 norm `scale`.
    """
    l2_val = float(tree_l2_norm(tree).item())
    safe_l2 = l2_val if l2_val > 0.0 else 1.0
    return tree_scalar_mul(float(scale) / safe_l2, tree)


def projection_l2_ball(
    tree: PyTree,
    scale: float = 1,
) -> PyTree:
    """Project `tree` onto the L2 ball `{y : ||y||_2 <= scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Radius of the L2 ball (default `1`).

    Returns:
        Projected PyTree with L2 norm at most `scale`.
    """
    l2_val = float(tree_l2_norm(tree).item())
    if l2_val <= float(scale):
        return tree
    return projection_l2_sphere(tree, scale=scale)


def _project_unit_simplex_1d(vals: Sequence[float]) -> list[float]:
    """Project a 1D sequence of floats onto the unit probability simplex `{p >= 0, sum(p) = 1}`.

    Args:
        vals: Input 1D sequence of floats.

    Returns:
        Projected list of floats summing to 1.0.
    """
    n = len(vals)
    if n == 0:
        return []
    u = sorted(vals, reverse=True)
    cssv = 0.0
    rho = 0
    theta = 0.0
    for idx, val in enumerate(u):
        cssv += val
        cand = (cssv - 1.0) / float(idx + 1)
        if val - cand > 0.0:
            rho = idx + 1
            theta = cand
    del rho
    return [max(v - theta, 0.0) for v in vals]


def projection_simplex(
    tree: PyTree,
    scale: Scalar | None = 1,
) -> PyTree:
    """Project `tree` onto the simplex `{p >= 0 : sum(p) = scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Target sum of the projected simplex (default `1`).

    Returns:
        Projected PyTree with non-negative elements summing to `scale`.

    Raises:
        ValueError: If `scale` is negative.
    """
    scale_val = 1.0 if scale is None else float(scale)
    if scale_val < 0.0:
        raise ValueError(f"scale must be non-negative, got {scale_val}.")
    if scale_val == 0.0:
        return tree_full_like(tree, 0.0)

    raw_leaves = tree_leaves(tree)
    flat_all: list[float] = []
    leaf_Metadata: list[tuple[tuple[int, ...], str] | None] = []
    for leaf in raw_leaves:
        if leaf is None:
            leaf_Metadata.append(None)
            continue
        t = asarray(leaf)
        vals = _flatten_tensor_floats(t)
        flat_all.extend(v / scale_val for v in vals)
        leaf_Metadata.append((t.shape, t.dtype.value))

    proj_unit = _project_unit_simplex_1d(flat_all)
    proj_scaled = [v * scale_val for v in proj_unit]

    rebuilt_leaves: list[TensorLeaf] = []
    offset = 0
    for meta in leaf_Metadata:
        if meta is None:
            rebuilt_leaves.append(None)
            continue
        shape, dt_str = meta
        count = 1
        for dim in shape:
            count *= dim
        chunk = proj_scaled[offset : offset + count]
        offset += count
        if shape == ():
            rebuilt_leaves.append(asarray(chunk[0], dtype=dt_str))
        else:
            rebuilt_leaves.append(reshape(asarray(chunk, dtype=dt_str), shape))

    out_tree, _ = _unflatten_like(tree, rebuilt_leaves, 0)
    return out_tree


def projection_l1_sphere(
    tree: PyTree,
    scale: float = 1,
) -> PyTree:
    """Project `tree` onto the L1 sphere `{y : ||y||_1 = scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Radius of the L1 sphere (default `1`).

    Returns:
        Projected PyTree with L1 norm equal to `scale`.
    """

    def _abs_leaf(x: TensorLeaf) -> TensorLeaf:
        """Compute `|x|` for leaf `x`.

        Args:
            x: Input leaf.

        Returns:
            Absolute value Tensor leaf.
        """
        return None if x is None else abs_tensor(asarray(x))

    def _sign_leaf(x: TensorLeaf) -> TensorLeaf:
        """Compute `sign(x)` for leaf `x`.

        Args:
            x: Input leaf.

        Returns:
            Sign Tensor leaf.
        """
        return None if x is None else sign(asarray(x))

    tree_abs = tree_map(_abs_leaf, tree)
    tree_sign = tree_map(_sign_leaf, tree)
    tree_abs_proj = projection_simplex(tree_abs, scale=scale)
    return tree_mul(tree_sign, tree_abs_proj)


def projection_l1_ball(
    tree: PyTree,
    scale: float = 1,
) -> PyTree:
    """Project `tree` onto the L1 ball `{y : ||y||_1 <= scale}`.

    Args:
        tree: Input PyTree to project.
        scale: Radius of the L1 ball (default `1`).

    Returns:
        Projected PyTree with L1 norm at most `scale`.
    """
    l1_val = float(tree_l1_norm(tree).item())
    if l1_val <= float(scale):
        return tree
    return projection_l1_sphere(tree, scale=scale)


def projection_vector(
    x: PyTree,
    a: PyTree,
) -> PyTree:
    """Project `x` onto the ray/line spanned by `a` (`(<x, a> / <a, a>) * a`).

    Args:
        x: Input PyTree to project.
        a: Direction PyTree defining the vector subspace.

    Returns:
        Projected PyTree along `a`.
    """
    denom = float(tree_vdot(a, a).item())
    safe_denom = denom if denom > 0.0 else 1.0
    factor = float(tree_vdot(x, a).item()) / safe_denom
    return tree_scalar_mul(factor, a)


def projection_hyperplane(
    x: PyTree,
    a: PyTree,
    b: Scalar,
) -> PyTree:
    """Project `x` onto the hyperplane `{y : <a, y> = b}`.

    Args:
        x: Input PyTree to project.
        a: Normal vector PyTree defining the hyperplane.
        b: Scalar offset of the hyperplane.

    Returns:
        Projected PyTree satisfying `<a, y> = b`.
    """
    denom = float(tree_vdot(a, a).item())
    safe_denom = denom if denom > 0.0 else 1.0
    scale_factor = (float(b) - float(tree_vdot(x, a).item())) / safe_denom
    return tree_add_scale(x, scale_factor, a)


def projection_halfspace(
    x: PyTree,
    a: PyTree,
    b: Scalar,
) -> PyTree:
    """Project `x` onto the halfspace `{y : <a, y> <= b}`.

    Args:
        x: Input PyTree to project.
        a: Normal vector PyTree defining the halfspace.
        b: Scalar upper bound of the halfspace.

    Returns:
        Projected PyTree satisfying `<a, y> <= b`.
    """
    denom = float(tree_vdot(a, a).item())
    safe_denom = denom if denom > 0.0 else 1.0
    diff = float(b) - float(tree_vdot(x, a).item())
    scale_factor = -max(-diff, 0.0) / safe_denom
    return tree_add_scale(x, scale_factor, a)


__all__ = [
    "projection_box",
    "projection_halfspace",
    "projection_hypercube",
    "projection_hyperplane",
    "projection_l1_ball",
    "projection_l1_sphere",
    "projection_l2_ball",
    "projection_l2_sphere",
    "projection_linf_ball",
    "projection_non_negative",
    "projection_simplex",
    "projection_vector",
]
