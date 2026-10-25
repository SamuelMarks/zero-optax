"""Rectified Adam (RAdam) optimizer."""

from __future__ import annotations

import math

from zero_optax._tree import (
    asarray,
    sqrt,
    tree_map,
    tree_zeros_like,
)
from zero_optax._types import (
    Params,
    ScalarOrSchedule,
    TensorLeaf,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    OptState,
)
from zero_optax.combine import chain
from zero_optax.transform import ScaleByAdamState, scale_by_learning_rate


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    *,
    nesterov: bool = False,
) -> GradientTransformation[ScaleByAdamState]:
    """Rescale updates according to the Rectified Adam (RAdam) algorithm.

    Args:
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        threshold: Tractability threshold on the approximated SMA length `rho_t`.
        nesterov: Whether to use Nesterov momentum for the first moment.

    Returns:
        A `GradientTransformation` implementing Rectified Adam scaling.
    """
    denom_b2 = 1.0 - b2
    ro_inf = (2.0 / (denom_b2 if abs(denom_b2) > 1e-12 else 1e-12)) - 1.0

    def init_fn(params: Params) -> ScaleByAdamState:
        """Initialize `ScaleByAdamState` with zero moments.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByAdamState`.
        """
        return ScaleByAdamState(
            count=asarray(0),
            mu=tree_zeros_like(params),
            nu=tree_zeros_like(params),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByAdamState]:
        """Update moments and apply variance rectification when `rho_t >= threshold`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `ScaleByAdamState`.
            params: Optional model parameters PyTree.

        Returns:
            Tuple of `(radam_updates, new_adam_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByAdamState`.
        """
        del params
        if not isinstance(state, ScaleByAdamState):
            raise TypeError(
                f"Expected ScaleByAdamState in scale_by_radam, got {type(state)!r}."
            )
        count_inc = state.count + 1
        t = int(count_inc.item())
        b1_t = b1**t
        b2_t = b2**t
        bc1 = 1.0 - b1_t
        bc2 = 1.0 - b2_t
        safe_bc1 = bc1 if abs(bc1) > 1e-12 else 1.0
        safe_bc2 = bc2 if abs(bc2) > 1e-12 else 1.0

        ro = ro_inf - 2.0 * float(t) * b2_t / safe_bc2

        def _update_mu(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update first moment `b1 * m + (1 - b1) * g`.

            Args:
                m: Previous first moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated first moment leaf.
            """
            if m is None or g is None:
                return m
            return b1 * asarray(m) + (1.0 - b1) * asarray(g)

        def _update_nu(v: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Update second moment `b2 * v + (1 - b2) * g^2`.

            Args:
                v: Previous second moment leaf.
                g: Current gradient leaf.

            Returns:
                Updated second moment leaf.
            """
            if v is None or g is None:
                return v
            g_t = asarray(g)
            return b2 * asarray(v) + (1.0 - b2) * (g_t * g_t)

        new_mu = tree_map(_update_mu, state.mu, updates)
        new_nu = tree_map(_update_nu, state.nu, updates)

        def _compute_mu_hat(m: TensorLeaf, g: TensorLeaf) -> TensorLeaf:
            """Compute bias-corrected (optionally Nesterov) first moment.

            Args:
                m: Updated first moment leaf.
                g: Current gradient leaf.

            Returns:
                Bias-corrected `mu_hat` leaf.
            """
            if m is None or g is None:
                return m
            m_t = asarray(m)
            if nesterov:
                bc_next = 1.0 - (b1 ** (t + 1))
                safe_next = bc_next if abs(bc_next) > 1e-12 else 1.0
                return (b1 * m_t / safe_next) + ((1.0 - b1) * asarray(g) / safe_bc1)
            return m_t / safe_bc1

        mu_hat = tree_map(_compute_mu_hat, new_mu, updates)

        if ro >= threshold:
            r_num = (ro - 4.0) * (ro - 2.0) * ro_inf
            r_den = (ro_inf - 4.0) * (ro_inf - 2.0) * ro
            rect = math.sqrt(max(r_num / r_den, 0.0))

            def _rect_step(mh: TensorLeaf, v: TensorLeaf) -> TensorLeaf:
                """Compute rectified adaptive update `rect * mh / (sqrt(v / bc2 + eps_root) + eps)`.

                Args:
                    mh: Bias-corrected first moment leaf.
                    v: Updated second moment leaf.

                Returns:
                    Rectified update leaf.
                """
                if mh is None or v is None:
                    return mh
                nu_hat = asarray(v) / safe_bc2
                return (rect * asarray(mh)) / (sqrt(nu_hat + eps_root) + eps)

            out_updates = tree_map(_rect_step, mu_hat, new_nu)
        else:
            out_updates = mu_hat

        return (
            out_updates,
            ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu),
        )

    return GradientTransformation(init_fn, update_fn)


def radam(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    *,
    nesterov: bool = False,
) -> GradientTransformation[tuple[OptState, ...]]:
    """The Rectified Adam (RAdam) optimizer.

    Args:
        learning_rate: Global scaling factor or learning rate `Schedule`.
        b1: Exponential decay rate for the first moment estimate.
        b2: Exponential decay rate for the second moment estimate.
        eps: Small constant added outside the square root in the denominator.
        eps_root: Small constant added inside the square root in the denominator.
        threshold: Threshold for variance tractability.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        A `GradientTransformation` implementing Rectified Adam.
    """
    return chain(
        scale_by_radam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            threshold=threshold,
            nesterov=nesterov,
        ),
        scale_by_learning_rate(learning_rate),
    )


__all__ = [
    "radam",
    "scale_by_radam",
]
