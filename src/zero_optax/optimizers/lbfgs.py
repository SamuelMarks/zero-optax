"""Limited-memory BFGS (L-BFGS) quasi-Newton optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zero_optax._linesearch import (
    ScaleByZoomLinesearchState,
    scale_by_zoom_linesearch,
)
from zero_optax._tree import (
    asarray,
    tree_zeros_like,
)
from zero_optax._types import (
    Params,
    PyTree,
    ScalarOrSchedule,
    Tensor,
    Updates,
)
from zero_optax.base import (
    GradientTransformation,
    GradientTransformationExtraArgs,
    OptState,
    _as_count_tensor,
    _OptStateDataclassBase,
    identity,
)
from zero_optax.combine import chain
from zero_optax.transform import scale_by_learning_rate
from zero_optax.tree_utils._tree_math import (
    tree_add_scale,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub,
    tree_vdot,
)


@dataclass(frozen=True)
class ScaleByLBFGSState(_OptStateDataclassBase):
    """State for `scale_by_lbfgs` holding iteration count, previous iterate/grad, and history pairs.

    Attributes:
        count: Optimization step counter Tensor.
        params: Previous model parameters PyTree.
        updates: Previous gradient updates PyTree.
        diff_params_memory: Ring buffer list of parameter differences `s_k = w_{k+1} - w_k`.
        diff_updates_memory: Ring buffer list of gradient differences `y_k = g_{k+1} - g_k`.
        weights_memory: Ring buffer list of scalar curvature reciprocals `rho_k = 1 / <y_k, s_k>`.
    """

    count: Tensor
    params: Params
    updates: Updates
    diff_params_memory: tuple[Params, ...]
    diff_updates_memory: tuple[Updates, ...]
    weights_memory: tuple[float, ...]
    _fields: tuple[str, ...] = (
        "count",
        "params",
        "updates",
        "diff_params_memory",
        "diff_updates_memory",
        "weights_memory",
    )

    def _asdict(self) -> dict[str, PyTree]:
        """Return state fields as an ordered dictionary.

        Returns:
            Dictionary of `ScaleByLBFGSState` fields.
        """
        return {
            "count": self.count,
            "params": self.params,
            "updates": self.updates,
            "diff_params_memory": self.diff_params_memory,
            "diff_updates_memory": self.diff_updates_memory,
            "weights_memory": self.weights_memory,
        }

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> ScaleByLBFGSState:
        """Construct `ScaleByLBFGSState` from a sequence of field values.

        Args:
            iterable: Sequence containing `(count, params, updates, diff_params_memory, diff_updates_memory, weights_memory)`.

        Returns:
            Reconstructed `ScaleByLBFGSState`.
        """
        items = list(iterable)
        dp_raw = items[3]
        du_raw = items[4]
        wm_raw = items[5]
        dp_mem = tuple(dp_raw) if isinstance(dp_raw, (tuple, list)) else ()
        du_mem = tuple(du_raw) if isinstance(du_raw, (tuple, list)) else ()
        wm_mem: list[float] = [
            float(w_item.item())
            if isinstance(w_item, Tensor)
            else float(w_item)
            if isinstance(w_item, (int, float))
            else 0.0
            for w_item in (wm_raw if isinstance(wm_raw, (tuple, list)) else ())
        ]
        return cls(
            count=_as_count_tensor(items[0]),
            params=items[1],
            updates=items[2],
            diff_params_memory=dp_mem,
            diff_updates_memory=du_mem,
            weights_memory=tuple(wm_mem),
        )


def scale_by_lbfgs(
    memory_size: int = 10,
    scale_init_precond: bool = True,
) -> GradientTransformation[ScaleByLBFGSState]:
    """Scale updates by the L-BFGS two-loop recursion inverse-Hessian approximation.

    Args:
        memory_size: Number of past `(s_k, y_k)` pairs to retain (`>= 1`).
        scale_init_precond: Whether to scale the initial identity preconditioner by
            `<s_{k-1}, y_{k-1}> / <y_{k-1}, y_{k-1}>` (or `1 / ||g_0||` on step 0).

    Returns:
        A `GradientTransformation` computing the L-BFGS preconditioned search direction.

    Raises:
        ValueError: If `memory_size < 1`.
    """
    if memory_size < 1:
        raise ValueError("memory_size must be >= 1.")

    def init_fn(params: Params) -> ScaleByLBFGSState:
        """Initialize `ScaleByLBFGSState` with empty history buffers.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `ScaleByLBFGSState`.
        """
        return ScaleByLBFGSState(
            count=asarray(0),
            params=tree_zeros_like(params),
            updates=tree_zeros_like(params),
            diff_params_memory=(),
            diff_updates_memory=(),
            weights_memory=(),
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, ScaleByLBFGSState]:
        """Update `(s_k, y_k)` history and run L-BFGS two-loop recursion on `updates`.

        Args:
            updates: Current gradient updates PyTree.
            state: Current `ScaleByLBFGSState`.
            params: Current model parameters PyTree (required).

        Returns:
            Tuple of `(preconditioned_updates, new_lbfgs_state)`.

        Raises:
            TypeError: If `state` is not a `ScaleByLBFGSState`.
            ValueError: If `params` is None.
        """
        if params is None:
            raise ValueError("`params` must be provided to `scale_by_lbfgs` update.")
        if not isinstance(state, ScaleByLBFGSState):
            raise TypeError(
                f"Expected ScaleByLBFGSState in scale_by_lbfgs, got {type(state)!r}."
            )
        step = int(state.count.item())
        s_mem = list(state.diff_params_memory)
        y_mem = list(state.diff_updates_memory)
        rho_mem = list(state.weights_memory)

        if step > 0:
            s_prev = tree_sub(params, state.params)
            y_prev = tree_sub(updates, state.updates)
            ys = float(tree_vdot(y_prev, s_prev).item())
            if abs(ys) > 1e-12:
                s_mem.append(s_prev)
                y_mem.append(y_prev)
                rho_mem.append(1.0 / ys)
                if len(s_mem) > memory_size:
                    s_mem = s_mem[-memory_size:]
                    y_mem = y_mem[-memory_size:]
                    rho_mem = rho_mem[-memory_size:]

        q = updates
        alphas: list[float] = []
        for s_i, y_i, rho_i in zip(reversed(s_mem), reversed(y_mem), reversed(rho_mem)):
            alpha_i = rho_i * float(tree_vdot(s_i, q).item())
            alphas.append(alpha_i)
            q = tree_add_scale(q, -alpha_i, y_i)
        alphas.reverse()

        if scale_init_precond:
            if s_mem:
                s_last = s_mem[-1]
                y_last = y_mem[-1]
                yy = float(tree_vdot(y_last, y_last).item())
                sy = float(tree_vdot(s_last, y_last).item())
                gamma = (sy / yy) if yy > 1e-12 else 1.0
            else:
                g_norm = float(tree_l2_norm(updates).item())
                gamma = min(1.0, 1.0 / g_norm) if g_norm > 0.0 else 1.0
        else:
            gamma = 1.0

        r = tree_scalar_mul(gamma, q)
        for s_i, y_i, rho_i, alpha_i in zip(s_mem, y_mem, rho_mem, alphas):
            beta_i = rho_i * float(tree_vdot(y_i, r).item())
            r = tree_add_scale(r, alpha_i - beta_i, s_i)

        return (
            r,
            ScaleByLBFGSState(
                count=state.count + 1,
                params=params,
                updates=updates,
                diff_params_memory=tuple(s_mem),
                diff_updates_memory=tuple(y_mem),
                weights_memory=tuple(rho_mem),
            ),
        )

    return GradientTransformation(init_fn, update_fn)


_DEFAULT_LBFGS_LINESEARCH: GradientTransformationExtraArgs[
    ScaleByZoomLinesearchState
] = scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy="one")


def lbfgs(
    learning_rate: ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    linesearch: (
        GradientTransformationExtraArgs[OptState]
        | GradientTransformation[OptState]
        | None
    ) = _DEFAULT_LBFGS_LINESEARCH,
) -> GradientTransformationExtraArgs[tuple[OptState, ...]]:
    """The Limited-memory BFGS (L-BFGS) quasi-Newton optimizer.

    Args:
        learning_rate: Optional global scaling factor or `Schedule`. If None, step size
            is determined by `linesearch` (or `1.0` if `linesearch is None`).
        memory_size: Number of past `(s_k, y_k)` curvature pairs to retain.
        scale_init_precond: Whether to scale the initial identity preconditioner.
        linesearch: Optional line-search transformation (defaults to zoom line search).

    Returns:
        A `GradientTransformationExtraArgs` implementing L-BFGS.
    """
    scaling_lr: ScalarOrSchedule = 1.0 if learning_rate is None else learning_rate
    ls_tx = identity() if linesearch is None else linesearch
    return chain(
        scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        scale_by_learning_rate(scaling_lr, flip_sign=True),
        ls_tx,
    )


__all__ = [
    "ScaleByLBFGSState",
    "lbfgs",
    "scale_by_lbfgs",
]
