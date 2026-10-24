"""Fromage (Frobenius matched gradient descent) optimizer."""

from __future__ import annotations

from typing import NamedTuple

from zero_optax._tree import (
    asarray,
    evaluate_schedule,
    safe_norm,
    sqrt,
    tree_map,
    where,
)
from zero_optax._types import Params, ScalarOrSchedule, Tensor, TensorLeaf, Updates
from zero_optax.base import GradientTransformation, OptState


class FromageState(NamedTuple):
    """State for the Fromage optimizer.

    Attributes:
        count: Optimization step counter Tensor.
    """

    count: Tensor


def fromage(
    learning_rate: ScalarOrSchedule,
    min_norm: float = 1e-6,
) -> GradientTransformation:
    """Create the Frobenius matched gradient descent (Fromage) optimizer.

    Args:
        learning_rate: Global scaling factor or schedule callable.
        min_norm: Minimum floor clipped onto both parameter and gradient norms.

    Returns:
        A `GradientTransformation` implementing Fromage.
    """

    def init_fn(params: Params) -> FromageState:
        """Initialize `FromageState` with `count = 0`.

        Args:
            params: Model parameters PyTree (unused).

        Returns:
            Initialized `FromageState`.
        """
        del params
        return FromageState(count=asarray(0))

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, FromageState]:
        """Compute Fromage trust-ratio update and norm-preserving weight decay.

        Args:
            updates: Gradient updates PyTree.
            state: Current `FromageState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(transformed_updates, new_fromage_state)`.

        Raises:
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `fromage` update.")
        count = state.count if isinstance(state, FromageState) else asarray(0)
        lr_t = asarray(evaluate_schedule(learning_rate, count))
        mult = 1.0 / sqrt(1.0 + lr_t * lr_t)
        eff_lr = -lr_t * mult
        decay_factor = mult - 1.0

        def _fromage_leaf(u: TensorLeaf, p: TensorLeaf) -> TensorLeaf:
            """Compute Fromage update for a single parameter leaf.

            Args:
                u: Gradient update leaf.
                p: Parameter leaf.

            Returns:
                Fromage update Tensor leaf.
            """
            if u is None or p is None:
                return u
            u_t = asarray(u)
            p_t = asarray(p)
            p_norm = safe_norm(p_t, min_norm)
            u_norm = safe_norm(u_t, min_norm)
            trust_ratio = p_norm / u_norm
            zero_norm = (p_norm == 0.0) | (u_norm == 0.0)
            safe_ratio = where(zero_norm, 1.0, trust_ratio)
            scaled_u = eff_lr * (u_t * safe_ratio)
            return scaled_u + decay_factor * p_t

        new_updates = tree_map(_fromage_leaf, updates, params)
        return new_updates, FromageState(count=count + 1)

    return GradientTransformation(init_fn, update_fn)


__all__ = [
    "FromageState",
    "fromage",
]
