"""SGD with Polyak step-size (`polyak_sgd` and `scale_by_polyak`)."""

from __future__ import annotations

from zero_optax._linesearch import _is_scalar_extra
from zero_optax._tree import (
    asarray,
)
from zero_optax._types import (
    Params,
    ScalarOrSchedule,
    Updates,
)
from zero_optax.base import (
    EmptyState,
    ExtraArgValue,
    GradientTransformationExtraArgs,
    OptState,
)
from zero_optax.combine import chain
from zero_optax.transform import scale_by_learning_rate
from zero_optax.tree_utils._tree_math import tree_l2_norm, tree_scalar_mul


def scale_by_polyak(
    max_learning_rate: float = 1.0,
    f_min: float = 0.0,
    eps: float = 0.0,
    variant: str = "sps",
) -> GradientTransformationExtraArgs[EmptyState]:
    """Scale updates by Polyak's step-size `min((f(x) - f_min) / (||g||^2 + eps), max_lr)`.

    Args:
        max_learning_rate: Maximum permissible step size.
        f_min: Lower bound on the objective function `f^*`.
        eps: Non-negative constant added to `||g||^2` in the denominator.
        variant: Either `'sps'` or `'sps+'` (which clamps `f(x) - f_min` below at 0).

    Returns:
        A `GradientTransformationExtraArgs` scaling updates by the Polyak step size.

    Raises:
        ValueError: If `variant` is not `'sps'` or `'sps+'`.
    """
    if variant not in ("sps", "sps+"):
        raise ValueError(
            f"Invalid argument for `variant`, expected 'sps' or 'sps+', got {variant!r}."
        )

    def init_fn(params: Params) -> EmptyState:
        """Initialize empty state for `scale_by_polyak`.

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
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, EmptyState]:
        """Scale `updates` by the Polyak step size using `value` from `extra_args`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `EmptyState`.
            params: Optional model parameters PyTree.
            **extra_args: Must include `value` holding the current objective scalar.

        Returns:
            Tuple of `(scaled_updates, EmptyState())`.

        Raises:
            ValueError: If `value` is not supplied in `extra_args`.
        """
        del state, params
        raw_val = extra_args.get("value")
        if not _is_scalar_extra(raw_val):
            raise ValueError(
                "`value` must be passed as a keyword argument to `polyak_sgd` / `scale_by_polyak` update."
            )
        f_val = float(asarray(raw_val).item())
        grad_sq = float(tree_l2_norm(updates, squared=True).item())
        gap = (
            (f_val - float(f_min))
            if variant == "sps"
            else max(f_val - float(f_min), 0.0)
        )
        denom = grad_sq + float(eps)
        step = min(gap / denom, float(max_learning_rate)) if denom > 0.0 else 0.0
        return tree_scalar_mul(step, updates), EmptyState()

    return GradientTransformationExtraArgs(init_fn, update_fn)


def polyak_sgd(
    max_learning_rate: float = 1.0,
    scaling: ScalarOrSchedule = 1.0,
    f_min: float = 0.0,
    eps: float = 0.0,
    variant: str = "sps",
) -> GradientTransformationExtraArgs[tuple[OptState, ...]]:
    """SGD with Polyak step-size.

    Args:
        max_learning_rate: Maximum permissible step size (default `1.0`).
        scaling: Global scaling factor or `Schedule` (default `1.0`).
        f_min: Lower bound on the objective function `f^*` (default `0.0`).
        eps: Non-negative constant added to `||g||^2` in denominator (default `0.0`).
        variant: Either `'sps'` or `'sps+'` (default `'sps'`).

    Returns:
        A `GradientTransformationExtraArgs` implementing Polyak SGD.
    """
    f_min_val = 0.0 if isinstance(f_min, str) else float(f_min)
    return chain(
        scale_by_polyak(
            max_learning_rate=max_learning_rate,
            f_min=f_min_val,
            eps=eps,
            variant=variant,
        ),
        scale_by_learning_rate(scaling),
    )


__all__ = [
    "polyak_sgd",
    "scale_by_polyak",
]
