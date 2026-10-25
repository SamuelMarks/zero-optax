"""Gradient clipping transformations."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import (
    asarray,
    leaf_l2_norm,
    maximum,
    minimum,
    tree_l2_norm,
    tree_map,
    where,
)
from zero_optax._types import Params, Tensor, TensorLeaf, Updates
from zero_optax.base import GradientTransformation, OptState


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


def adaptive_grad_clip(
    clipping: float,
    eps: float = 1e-3,
) -> GradientTransformation[AdaptiveGradClipState]:
    """Clip updates adaptively based on the ratio of gradient norm to parameter norm.

    Args:
        clipping: Maximum allowed ratio of gradient norm to parameter norm.
        eps: Minimum lower bound on parameter norm to avoid zero-division.

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
            p_norm = maximum(leaf_l2_norm(p), eps)
            g_norm = leaf_l2_norm(u_t)
            max_norm = p_norm * clipping
            safe_g = where(g_norm == 0.0, 1.0, g_norm)
            return where(g_norm > max_norm, u_t * (max_norm / safe_g), u_t)

        return tree_map(_agc_leaf, updates, params), AdaptiveGradClipState()

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "AdaptiveGradClipState",
    "ClipByGlobalNormState",
    "ClipState",
    "adaptive_grad_clip",
    "clip",
    "clip_by_global_norm",
]
