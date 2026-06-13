"""Adam optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, OptState, Params, Updates
from zero_optax.optimizers.sgd import ScaleByScheduleState


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: jnp.ndarray
    mu: Updates
    nu: Updates


class AdamState(NamedTuple):
    """State for the Adam optimizer."""

    adam_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def adam(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation:
    """The Adam optimizer.

    Adam is an SGD variant with gradient scaling adaptation. The scaling
    used for each parameter is computed from estimates of first and second-order
    moments of the gradients (using suitable exponential moving averages).

    Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
      b1: Exponential decay rate to track the first moment of past gradients.
      b2: Exponential decay rate to track the second moment of past gradients.
      eps: A small constant applied to denominator outside of the square root
        (as in the Adam paper) to avoid dividing by zero when rescaling.
      eps_root: A small constant applied to denominator inside the square root (as
        in RMSProp), to avoid dividing by zero when rescaling. This is needed for
        example when computing (meta-)gradients through Adam.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> AdamState:
        mu = tree_map(
            lambda t: jnp.zeros_like(
                t, dtype=mu_dtype if mu_dtype is not None else t.dtype
            ),
            params,
        )
        nu = tree_map(lambda t: jnp.zeros_like(t), params)
        return AdamState(
            adam_state=ScaleByAdamState(count=jnp.array(0), mu=mu, nu=nu),
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: AdamState, params: Optional[Params] = None
    ) -> Tuple[Updates, AdamState]:
        count_inc = state.adam_state.count + 1
        mu = state.adam_state.mu
        nu = state.adam_state.nu

        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, mu, updates)
        new_nu = tree_map(lambda n, g: b2 * n + (1 - b2) * jnp.square(g), nu, updates)

        b1_t = b1**count_inc
        b2_t = b2**count_inc

        def _compute_update(m, n, g):
            if nesterov:
                m_hat = (b1 * m + (1 - b1) * g) / (1 - b1_t)
            else:
                m_hat = m / (1 - b1_t)
            n_hat = n / (1 - b2_t)
            return m_hat / (jnp.sqrt(n_hat + eps_root) + eps)

        new_updates = tree_map(_compute_update, new_mu, new_nu, updates)

        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        new_updates = tree_map(lambda u: -step_lr * u, new_updates)

        return new_updates, AdamState(
            adam_state=ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)
