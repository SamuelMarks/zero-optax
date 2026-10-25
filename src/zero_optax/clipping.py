"""Gradient clipping transformations."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import (
    asarray,
    maximum,
    minimum,
    reduce_mean,
    reduce_sum,
    sqrt,
    tree_l2_norm,
    tree_leaves,
    tree_map,
    where,
)
from zero_optax._types import Params, PyTree, Tensor, TensorLeaf, Updates
from zero_optax.base import EmptyState, GradientTransformation, OptState


class ClipState(NamedTuple):
    """Empty state for element-wise `clip` transformation."""


class ClipByGlobalNormState(NamedTuple):
    """Empty state for `clip_by_global_norm` transformation."""


class AdaptiveGradClipState(NamedTuple):
    """Empty state for `adaptive_grad_clip` transformation."""


def clip(max_delta: float | Tensor) -> GradientTransformation[ClipState]:
    """Clip gradient updates element-wise to `[-max_delta, +max_delta]`.

    Args:
        max_delta: Maximum absolute value for any element in the updates PyTree.

    Returns:
        A `GradientTransformation` performing element-wise clipping.
    """
    delta_t = asarray(max_delta)

    def init_fn(params: Params) -> ClipState:
        """Initialize empty `ClipState`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `ClipState`.
        """
        del params
        return ClipState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ClipState]:
        """Clip each leaf element-wise to `[-max_delta, +max_delta]`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ClipState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(clipped_updates, ClipState())`.
        """
        del state, params

        def _clip_leaf(u: TensorLeaf) -> TensorLeaf:
            """Clip a single leaf tensor to `[-delta_t, +delta_t]`.

            Args:
                u: Gradient update leaf.

            Returns:
                Clipped Tensor leaf.
            """
            if u is None:
                return None
            return minimum(maximum(asarray(u), -delta_t), delta_t)

        return tree_map(_clip_leaf, updates), ClipState()

    return GradientTransformation(init_fn, update_fn)


def clip_by_global_norm(
    max_norm: float | Tensor,
) -> GradientTransformation[ClipByGlobalNormState]:
    """Clip gradient updates using their global L2 norm across the PyTree.

    Args:
        max_norm: Maximum permissible global L2 norm.

    Returns:
        A `GradientTransformation` that scales updates when global norm exceeds `max_norm`.
    """
    max_norm_t = asarray(max_norm)

    def init_fn(params: Params) -> ClipByGlobalNormState:
        """Initialize empty `ClipByGlobalNormState`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `ClipByGlobalNormState`.
        """
        del params
        return ClipByGlobalNormState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ClipByGlobalNormState]:
        """Scale updates down if their global L2 norm exceeds `max_norm`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ClipByGlobalNormState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(clipped_updates, ClipByGlobalNormState())`.
        """
        del state, params
        g_norm = tree_l2_norm(updates)
        trigger = g_norm < max_norm_t

        def _scale_leaf(u: TensorLeaf) -> TensorLeaf:
            """Scale leaf `u` when `g_norm >= max_norm_t`.

            Args:
                u: Gradient update leaf.

            Returns:
                Scaled Tensor leaf.
            """
            if u is None:
                return None
            t = asarray(u)
            safe_g = where(g_norm == 0.0, 1.0, g_norm)
            return where(trigger, t, (t / safe_g) * max_norm_t)

        return tree_map(_scale_leaf, updates), ClipByGlobalNormState()

    return GradientTransformation(init_fn, update_fn)


def _unitwise_norm(
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    """Compute the unit-wise L2 norm of a tensor along `axis` or default dimensions.

    Args:
        x: Input tensor.
        axis: Optional explicit axis or axes to reduce over with `keepdims=True`.

    Returns:
        Unit-wise L2 norm Tensor broadcastable to `x.shape`.
    """
    if axis is not None:
        return sqrt(reduce_sum(x * x, axis=axis, keepdims=True))
    if x.ndim <= 1:
        return sqrt(reduce_sum(x * x, keepdims=False))
    reduce_axes = tuple(range(x.ndim - 1))
    return sqrt(reduce_sum(x * x, axis=reduce_axes, keepdims=True))


def adaptive_grad_clip(
    clipping: float,
    eps: float = 0.001,
    axis: int | tuple[int, ...] | None = None,
) -> GradientTransformation[AdaptiveGradClipState]:
    """Clip updates adaptively based on the ratio of gradient norm to parameter norm.

    Args:
        clipping: Maximum allowed ratio of gradient norm to parameter norm.
        eps: Minimum lower bound on parameter norm to avoid zero-division.
        axis: Optional axis or axes along which to compute the unit-wise norm.

    Returns:
        A `GradientTransformation` implementing Adaptive Gradient Clipping (AGC).
    """

    def init_fn(params: Params) -> AdaptiveGradClipState:
        """Initialize empty `AdaptiveGradClipState`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `AdaptiveGradClipState`.
        """
        del params
        return AdaptiveGradClipState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, AdaptiveGradClipState]:
        """Adaptively clip each gradient leaf relative to its parameter leaf norm.

        Args:
            updates: Gradient updates PyTree.
            state: Current `AdaptiveGradClipState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(clipped_updates, AdaptiveGradClipState())`.

        Raises:
            ValueError: If `params` is None.
        """
        del state
        if params is None:
            raise ValueError(
                "`params` must be provided to `adaptive_grad_clip` update."
            )

        def _agc_leaf(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Clip gradient leaf `u` against parameter leaf `p`.

            Args:
                u: Gradient update leaf.
                p: Parameter leaf.

            Returns:
                Adaptively clipped Tensor leaf.
            """
            if u is None or p is None:
                return u
            u_t = asarray(u)
            p_t = asarray(p)
            p_norm = maximum(_unitwise_norm(p_t, axis=axis), eps)
            g_norm = _unitwise_norm(u_t, axis=axis)
            max_norm = p_norm * clipping
            safe_g = where(g_norm == 0.0, 1.0, g_norm)
            return where(g_norm > max_norm, u_t * (max_norm / safe_g), u_t)

        return tree_map(_agc_leaf, updates, params), AdaptiveGradClipState()

    return GradientTransformation(init_fn, update_fn)


def clip_by_block_rms(threshold: float) -> GradientTransformation[EmptyState]:
    """Clip updates to a max root-mean-square for each parameter block/leaf.

    Args:
        threshold: Maximum root-mean-square threshold per tensor leaf.

    Returns:
        A `GradientTransformation` clipping each leaf by its RMS.
    """

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `clip_by_block_rms`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `EmptyState`.
        """
        del params
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, EmptyState]:
        """Clip each update leaf so its RMS does not exceed `threshold`.

        Args:
            updates: Gradient updates PyTree.
            state: Current optimizer state.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(clipped_updates, EmptyState())`.
        """
        del state, params

        def _clip_rms_leaf(u: TensorLeaf) -> TensorLeaf:
            """Clip single leaf `u` by its RMS.

            Args:
                u: Gradient update leaf.

            Returns:
                Clipped Tensor leaf.
            """
            if u is None:
                return None
            u_t = asarray(u)
            rms = sqrt(reduce_mean(u_t * u_t))
            safe_rms = where(rms == 0.0, 1.0, rms)
            denom = maximum(1.0, safe_rms / threshold)
            return where(rms == 0.0, u_t, u_t / denom)

        return tree_map(_clip_rms_leaf, updates), EmptyState()

    return GradientTransformation(init_fn, update_fn)


def per_example_global_norm_clip(
    grads: PyTree,
    l2_norm_clip: float,
) -> tuple[PyTree, Tensor]:
    """Clip per-example gradients along axis 0 by their global L2 norm and sum across examples.

    Args:
        grads: PyTree of batched gradients where each leaf has leading dimension `[batch_size, ...]`.
        l2_norm_clip: Maximum permissible L2 norm per example (`>= 0`).

    Returns:
        Tuple `(sum_clipped_grads, num_clipped)` where `sum_clipped_grads` is a PyTree of summed
        clipped gradients and `num_clipped` is a scalar integer Tensor counting clipped examples.

    Raises:
        ValueError: If `l2_norm_clip < 0`.
    """
    if l2_norm_clip < 0.0:
        raise ValueError(f"l2_norm_clip must be >= 0, got {l2_norm_clip}.")
    leaves = [asarray(leaf) for leaf in tree_leaves(grads) if leaf is not None]
    if not leaves:
        return grads, asarray(0, dtype="int32")
    batch_size = leaves[0].shape[0]
    sq_norms = [0.0] * batch_size
    for leaf_t in leaves:
        for b_idx in range(batch_size):
            row = leaf_t[b_idx]
            sq_norms[b_idx] += float(reduce_sum(row * row).item())

    scales: list[float] = []
    num_clipped = 0
    for sq in sq_norms:
        norm_i = float(sqrt(asarray(sq)).item())
        if norm_i > l2_norm_clip:
            num_clipped += 1
            scales.append(l2_norm_clip / norm_i if norm_i > 0.0 else 1.0)
        else:
            scales.append(1.0)

    def _clip_and_sum_leaf(g: TensorLeaf) -> TensorLeaf:
        """Scale each example slice `g[b]` by `scales[b]` and sum over axis 0.

        Args:
            g: Batched gradient leaf of shape `[batch_size, ...]`.

        Returns:
            Summed clipped gradient Tensor leaf of shape `[...]`.
        """
        if g is None:
            return None
        g_t = asarray(g)
        acc = g_t[0] * scales[0]
        for b_idx in range(1, batch_size):
            acc = acc + g_t[b_idx] * scales[b_idx]
        return acc

    return tree_map(_clip_and_sum_leaf, grads), asarray(num_clipped, dtype="int32")


__all__ = [
    "AdaptiveGradClipState",
    "ClipByGlobalNormState",
    "ClipState",
    "adaptive_grad_clip",
    "clip",
    "clip_by_block_rms",
    "clip_by_global_norm",
    "per_example_global_norm_clip",
]
