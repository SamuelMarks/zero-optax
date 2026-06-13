"""LARS optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, PyTree, Params, Updates
from zero_optax.optimizers.sgd import ScaleByScheduleState, TraceState


class LarsState(NamedTuple):
    """State for the LARS optimizer."""

    trace_state: TraceState
    scale_state: ScaleByScheduleState


def lars(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = True,
    trust_coefficient: float = 0.001,
    eps: float = 0.0,
    trust_ratio_mask: Optional[Union[PyTree, Callable[[PyTree], PyTree]]] = True,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> GradientTransformation:
    """The LARS optimizer.

    LARS is a layer-wise adaptive optimizer introduced to help scale SGD to
    larger batch sizes. LARS later inspired the LAMB optimizer.

    Args:
      learning_rate: A global scaling factor.
      weight_decay: Strength of the weight decay regularization.
      weight_decay_mask: A tree with same structure as (or a prefix of) the params
        PyTree, or a Callable that returns such a pytree given the params/updates.
      trust_coefficient: A multiplier for the trust ratio.
      eps: Optional additive constant in the trust ratio denominator.
      trust_ratio_mask: A tree with same structure as (or a prefix of) the params
        PyTree, or a Callable that returns such a pytree given the params/updates.
      momentum: Decay rate for momentum.
      nesterov: Whether to use Nesterov momentum.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> LarsState:
        trace = tree_map(lambda t: jnp.zeros_like(t), params)
        return LarsState(
            trace_state=TraceState(trace=trace),
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: LarsState, params: Optional[Params] = None
    ) -> Tuple[Updates, LarsState]:
        if params is None:
            raise ValueError(
                "You must pass `params` to the `update` function of `lars`."
            )

        # Determine masks
        if callable(weight_decay_mask):
            wd_mask = weight_decay_mask(params)
        elif isinstance(weight_decay_mask, bool):
            wd_mask = tree_map(lambda _: weight_decay_mask, params)
        else:
            wd_mask = weight_decay_mask

        if callable(trust_ratio_mask):
            tr_mask = trust_ratio_mask(params)
        elif isinstance(trust_ratio_mask, bool):
            tr_mask = tree_map(lambda _: trust_ratio_mask, params)
        else:
            tr_mask = trust_ratio_mask

        # Add decayed weights
        def _add_decayed_weights(g, p, msk):
            if msk:
                return g + weight_decay * p
            return g

        updates = tree_map(_add_decayed_weights, updates, params, wd_mask)

        # Trust ratio scaling
        def _scale_by_trust_ratio(g, p, msk):
            if not msk:
                return g
            param_norm = jnp.sqrt(jnp.sum(jnp.square(p)))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

            zero_norm = (param_norm == 0.0) + (update_norm == 0.0) > 0
            safe_trust_ratio = jnp.where(
                zero_norm, jnp.array(1.0, dtype=p.dtype), trust_ratio
            )
            return g * safe_trust_ratio

        updates = tree_map(_scale_by_trust_ratio, updates, params, tr_mask)

        # Learning rate
        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        updates = tree_map(lambda u: -step_lr * u, updates)

        # Trace
        trace = state.trace_state.trace

        def _update_momentum(g, t):
            return g + momentum * t

        new_trace = tree_map(_update_momentum, updates, trace)

        if nesterov:
            updates = tree_map(_update_momentum, updates, new_trace)
        else:
            updates = new_trace

        return updates, LarsState(
            trace_state=TraceState(trace=new_trace),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)
