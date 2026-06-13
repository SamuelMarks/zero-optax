"""AdamW optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, PyTree, Params, Updates
from zero_optax.optimizers.sgd import ScaleByScheduleState
from zero_optax.optimizers.adam import ScaleByAdamState


class AdamwState(NamedTuple):
    """State for the AdamW optimizer."""

    adam_state: ScaleByAdamState
    scale_state: ScaleByScheduleState


def adamw(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0001,
    mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation:
    """Adam with weight decay regularization.

    AdamW uses weight decay to regularize learning towards small weights, as
    this leads to better generalization.

    Args:
      learning_rate: A global scaling factor.
      b1: Exponential decay rate to track the first moment of past gradients.
      b2: Exponential decay rate to track the second moment of past gradients.
      eps: A small constant applied to denominator outside of the square root.
      eps_root: A small constant applied to denominator inside the square root.
      mu_dtype: Optional `dtype` to be used for the first order accumulator.
      weight_decay: Strength of the weight decay regularization.
      mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the weight decay to, and `False` for those you want to skip.
      nesterov: Whether to use Nesterov momentum.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> AdamwState:
        mu = tree_map(
            lambda t: jnp.zeros_like(
                t, dtype=mu_dtype if mu_dtype is not None else t.dtype
            ),
            params,
        )
        nu = tree_map(lambda t: jnp.zeros_like(t), params)
        return AdamwState(
            adam_state=ScaleByAdamState(count=jnp.array(0), mu=mu, nu=nu),
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: AdamwState, params: Optional[Params] = None
    ) -> Tuple[Updates, AdamwState]:
        if params is None:
            raise ValueError(
                "You must pass `params` to the `update` function of `adamw`."
            )

        count_inc = state.adam_state.count + 1
        mu = state.adam_state.mu
        nu = state.adam_state.nu

        new_mu = tree_map(lambda m, g: b1 * m + (1 - b1) * g, mu, updates)
        new_nu = tree_map(lambda n, g: b2 * n + (1 - b2) * jnp.square(g), nu, updates)

        b1_t = b1**count_inc
        b2_t = b2**count_inc

        # Determine mask
        if callable(mask):
            computed_mask = mask(params)
        elif isinstance(mask, bool):
            computed_mask = tree_map(lambda _: mask, params)
        elif mask is not None:
            computed_mask = mask
        else:
            computed_mask = tree_map(lambda _: True, params)

        def _compute_update(m, n, g, p, msk):
            if nesterov:
                m_hat = (b1 * m + (1 - b1) * g) / (1 - b1_t)
            else:
                m_hat = m / (1 - b1_t)
            n_hat = n / (1 - b2_t)
            u = m_hat / (jnp.sqrt(n_hat + eps_root) + eps)
            if msk:
                u = u + weight_decay * p
            return u

        new_updates = tree_map(
            _compute_update, new_mu, new_nu, updates, params, computed_mask
        )

        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        new_updates = tree_map(lambda u: -step_lr * u, new_updates)

        return new_updates, AdamwState(
            adam_state=ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)
