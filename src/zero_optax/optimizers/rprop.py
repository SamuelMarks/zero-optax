"""Resilient backpropagation (Rprop) and SM3 optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from zero_optax._tree import (
    asarray,
    full_like,
    maximum,
    minimum,
    sign,
    sqrt,
    tree_map,
    tree_zeros_like,
    where,
)
from zero_optax._types import (
    Params,
    PyTree,
    ScalarOrSchedule,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
    _OptStateDataclassBase,
)
from zero_optax.combine import chain
from zero_optax.transform import scale, scale_by_learning_rate


@dataclass(frozen=True)
class ScaleByRpropState(_OptStateDataclassBase):
    """State for `scale_by_rprop` holding adaptive step sizes and previous gradients.

    Attributes:
        step_sizes: Per-parameter adaptive step sizes PyTree.
        prev_updates: Previous step gradients PyTree.
    """

    step_sizes: Updates
    prev_updates: Updates
    _fields: tuple[str, ...] = ("step_sizes", "prev_updates")

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary with `'step_sizes'` and `'prev_updates'`.
        """
        return {
            "step_sizes": self.step_sizes,
            "prev_updates": self.prev_updates,
        }


def scale_by_rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> GradientTransformation[ScaleByRpropState]:
    """Scale updates with the Rprop (resilient backpropagation) rule.

    Args:
        learning_rate: Initial step size for all parameters.
        eta_minus: Multiplicative decrease factor (`< 1`) when gradient changes sign.
        eta_plus: Multiplicative increase factor (`> 1`) when gradient keeps the same sign.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.

    Returns:
        A `GradientTransformation` implementing Rprop step-size adaptation.
    """

    def init_fn(params: Params) -> ScaleByRpropState:
        """Initialize `ScaleByRpropState` with `learning_rate` step sizes and zero `prev_updates`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByRpropState`.
        """

        def _init_step(p: TensorLeaf) -> TensorLeaf:
            """Fill parameter leaf shape with `learning_rate`.

            Args:
                p: Parameter leaf.

            Returns:
                Constant `learning_rate` Tensor leaf.
            """
            return None if p is None else full_like(p, float(learning_rate))

        return ScaleByRpropState(
            step_sizes=tree_map(_init_step, params),
            prev_updates=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByRpropState]:
        """Adapt per-parameter step sizes based on sign agreement and return `-step_size * sign(g)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByRpropState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(rprop_updates, new_rprop_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByRpropState`.
        """
        del params
        if not isinstance(state, ScaleByRpropState):
            raise TypeError(
                f"Expected ScaleByRpropState in scale_by_rprop, got {type(state)!r}."
            )

        def _adapt_step(
            prev_step: TensorLeaf,
            g: TensorLeaf,
            prev_g: TensorLeaf,
        ) -> TensorLeaf:
            """Update step size leaf based on `g * prev_g` sign.

            Args:
                prev_step: Previous step size leaf.
                g: Current gradient leaf.
                prev_g: Previous gradient leaf.

            Returns:
                Clipped updated step size leaf.
            """
            if prev_step is None or g is None or prev_g is None:
                return prev_step
            prod = asarray(g) * asarray(prev_g)
            st_t = asarray(prev_step)
            inc = minimum(st_t * eta_plus, max_step_size)
            dec = maximum(st_t * eta_minus, min_step_size)
            return where(prod > 0.0, inc, where(prod < 0.0, dec, st_t))

        def _next_prev_g(
            g: TensorLeaf,
            prev_g: TensorLeaf,
        ) -> TensorLeaf:
            """Zero out stored gradient where sign flipped (`g * prev_g < 0`).

            Args:
                g: Current gradient leaf.
                prev_g: Previous gradient leaf.

            Returns:
                Stored gradient leaf for the next iteration.
            """
            if g is None or prev_g is None:
                return g
            g_t = asarray(g)
            prod = g_t * asarray(prev_g)
            return where(prod < 0.0, 0.0, g_t)

        def _compute_upd(step_sz: TensorLeaf, next_g: TensorLeaf) -> TensorLeaf:
            """Compute `step_size * sign(next_g)`.

            Args:
                step_sz: Updated step size leaf.
                next_g: Masked current gradient leaf.

            Returns:
                Scaled Rprop update leaf.
            """
            if step_sz is None or next_g is None:
                return next_g
            return asarray(step_sz) * sign(asarray(next_g))

        new_step_sizes = tree_map(
            _adapt_step, state.step_sizes, updates, state.prev_updates
        )
        new_prev_updates = tree_map(_next_prev_g, updates, state.prev_updates)
        out_updates = tree_map(_compute_upd, new_step_sizes, new_prev_updates)
        return (
            out_updates,
            ScaleByRpropState(
                step_sizes=new_step_sizes,
                prev_updates=new_prev_updates,
            ),
        )

    return GradientTransformation(init_fn, update_fn)


def rprop(
    learning_rate: ScalarOrSchedule,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The Rprop (resilient backpropagation) optimizer.

    Args:
        learning_rate: Initial step size or learning rate `Schedule`.
        eta_minus: Multiplicative factor (`< 1`) for decreasing step size on sign change.
        eta_plus: Multiplicative factor (`> 1`) for increasing step size on sign agreement.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.

    Returns:
        A `GradientTransformation` implementing Rprop.
    """
    if callable(learning_rate):
        return chain(
            scale_by_rprop(
                1.0,
                eta_minus=eta_minus,
                eta_plus=eta_plus,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
            ),
            scale_by_learning_rate(learning_rate),
        )
    return chain(
        scale_by_rprop(
            float(learning_rate),
            eta_minus=eta_minus,
            eta_plus=eta_plus,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
        ),
        scale(-1.0),
    )


class ScaleBySm3State(NamedTuple):
    """State for `scale_by_sm3` holding momentum `mu` and accumulator `nu`.

    Attributes:
        mu: Momentum accumulator PyTree.
        nu: Second-moment cover accumulator PyTree.
    """

    mu: Updates
    nu: Updates


def scale_by_sm3(
    b1: float = 0.9,
    b2: float = 1.0,
    eps: float = 1e-8,
) -> GradientTransformation[ScaleBySm3State]:
    """Scale updates by `sm3` (Square-root of Minima of Sums of Maxima of Squared-gradients).

    Args:
        b1: Momentum decay rate in `[0, 1)`.
        b2: Second-moment accumulation weight (1.0 for standard SM3).
        eps: Small constant added to denominator to prevent division by zero.

    Returns:
        A `GradientTransformation` implementing SM3 scaling.
    """

    def init_fn(params: Params) -> ScaleBySm3State:
        """Initialize `ScaleBySm3State` with zero `mu` and `nu`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleBySm3State`.
        """
        return ScaleBySm3State(
            mu=tree_zeros_like(params),
            nu=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleBySm3State]:
        """Update SM3 accumulators and scale updates.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleBySm3State`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(sm3_updates, new_sm3_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleBySm3State`.
        """
        del params
        if not isinstance(state, ScaleBySm3State):
            raise TypeError(
                f"Expected ScaleBySm3State in scale_by_sm3, got {type(state)!r}."
            )

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update SM3 second-moment estimate `b2 * v + (1 - b2_eff) * g^2`.

            Args:
                v: Previous accumulator leaf.
                g: Current gradient leaf.

            Returns:
                Updated accumulator leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            v_t = asarray(v)
            weight = 1.0 if b2 == 1.0 else (1.0 - b2)
            return b2 * v_t + weight * (g_t * g_t)

        def _scale_and_mu(
            m: TensorLeaf,
            v_new: TensorLeaf,
            g: TensorLeaf,
        ) -> TensorLeaf:
            """Compute SM3 step `b1 * m + (1 - b1) * (g / sqrt(v_new + eps))`.

            Args:
                m: Previous momentum leaf.
                v_new: Updated second-moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated momentum / step leaf.
            """
            if m is None or v_new is None or g is None:
                return g
            v_t = asarray(v_new)
            g_t = asarray(g)
            scaled = where(v_t > 0.0, g_t / sqrt(v_t + eps), 0.0)
            if b1 > 0.0:
                return b1 * asarray(m) + (1.0 - b1) * scaled
            return scaled

        new_nu = tree_map(_update_nu, state.nu, updates)
        new_mu = tree_map(_scale_and_mu, state.mu, new_nu, updates)
        return new_mu, ScaleBySm3State(mu=new_mu, nu=new_nu)

    return GradientTransformation(init_fn, update_fn)


def sm3(
    learning_rate: float,
    momentum: float = 0.9,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients) optimizer.

    Args:
        learning_rate: Global scaling factor (`step_size`).
        momentum: Momentum decay rate (default `0.9`).

    Returns:
        A `GradientTransformation` implementing SM3.
    """
    return chain(
        scale_by_sm3(b1=momentum),
        scale(-float(learning_rate)),
    )


__all__ = [
    "ScaleByRpropState",
    "ScaleBySm3State",
    "rprop",
    "scale_by_rprop",
    "scale_by_sm3",
    "sm3",
]
