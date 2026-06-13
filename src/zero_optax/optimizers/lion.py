"""Lion optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, PyTree, Params, Updates
from zero_optax.optimizers.sgd import ScaleByScheduleState


class ScaleByLionState(NamedTuple):
    """State for the Lion algorithm."""

    mu: Updates


class LionState(NamedTuple):
    """State for the Lion optimizer."""

    lion_state: ScaleByLionState
    scale_state: ScaleByScheduleState


def lion(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.001,
    mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = None,
) -> GradientTransformation:
    """The Lion optimizer.

    Lion is discovered by symbolic program search. Unlike most adaptive optimizers
    such as AdamW, Lion only tracks momentum, making it more memory-efficient.

    Args:
      learning_rate: A global scaling factor.
      b1: Rate to combine the momentum and the current gradient.
      b2: Exponential decay rate to track the momentum of past gradients.
      mu_dtype: Optional `dtype` to be used for the first order accumulator.
      weight_decay: Strength of the weight decay regularization.
      mask: A tree with same structure as (or a prefix of) the params PyTree, or a
        Callable that returns such a pytree given the params/updates.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> LionState:
        mu = tree_map(
            lambda t: jnp.zeros_like(
                t, dtype=mu_dtype if mu_dtype is not None else t.dtype
            ),
            params,
        )
        return LionState(
            lion_state=ScaleByLionState(mu=mu),
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: LionState, params: Optional[Params] = None
    ) -> Tuple[Updates, LionState]:
        if params is None:
            raise ValueError(
                "You must pass `params` to the `update` function of `lion`."
            )

        mu = state.lion_state.mu

        # Determine mask
        if callable(mask):
            computed_mask = mask(params)
        elif isinstance(mask, bool):
            computed_mask = tree_map(lambda _: mask, params)
        elif mask is not None:
            computed_mask = mask
        else:
            computed_mask = tree_map(lambda _: True, params)

        # Since tree_map doesn't easily return multiple trees, do two passes
        new_updates = tree_map(
            lambda m, g, p, msk: (
                jnp.sign(b1 * m + (1 - b1) * g) + (weight_decay * p if msk else 0.0)
            ),
            mu,
            updates,
            params,
            computed_mask,
        )
        new_mu = tree_map(lambda m, g: b2 * m + (1 - b2) * g, mu, updates)

        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        new_updates = tree_map(lambda u: -step_lr * u, new_updates)

        return new_updates, LionState(
            lion_state=ScaleByLionState(mu=new_mu),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)
