"""Adagrad optimizer."""

from typing import Callable, NamedTuple, Optional, Tuple, Union

import ml_switcheroo.jnp as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, OptState, Params, Updates
from zero_optax.optimizers.sgd import ScaleByScheduleState


class ScaleByRssState(NamedTuple):
    """State for Adagrad containing sum of squares."""

    sum_of_squares: Updates


class AdagradState(NamedTuple):
    """State for the Adagrad optimizer."""

    rss_state: ScaleByRssState
    scale_state: ScaleByScheduleState


def adagrad(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-07,
) -> GradientTransformation:
    """The Adagrad optimizer.

    AdaGrad is a sub-gradient algorithm for stochastic optimization that adapts
    the learning rate individually for each feature based on its gradient history.

    The updated parameters adopt the form:

    .. math::

      w_{t+1}^{(i)} = w_{t}^{(i)} - \\eta \\frac{g_{t}^{(i)}}
                   {\\sqrt{\\sum_{\\tau=1}^{t} (g_{\\tau}^{(i)})^2 + \\epsilon}}

    where:
      - :math:`w_t^{(i)}` is the parameter :math:`i` at time step :math:`t`,
      - :math:`\\eta` is the learning rate,
      - :math:`g_t^{(i)}` is the gradient of parameter :math:`i` at time step
        :math:`t`,
      - :math:`\\epsilon` is a small constant to ensure numerical stability.

    Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
      initial_accumulator_value: Initial value for the accumulator.
      eps: A small constant applied to denominator inside of the square root (as
        in RMSProp) to avoid dividing by zero when rescaling.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> AdagradState:
        sum_of_squares = tree_map(
            lambda t: jnp.full_like(t, initial_accumulator_value), params
        )
        return AdagradState(
            rss_state=ScaleByRssState(sum_of_squares=sum_of_squares),
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: AdagradState, params: Optional[Params] = None
    ) -> Tuple[Updates, AdagradState]:
        sum_of_squares = state.rss_state.sum_of_squares

        # Since tree_map doesn't easily return multiple trees, we do it in two passes
        # or we can do it safely since they share structure.
        new_updates = tree_map(
            lambda g, sq: g / jnp.sqrt(sq + jnp.square(g) + eps),
            updates,
            sum_of_squares,
        )
        new_sum_of_squares = tree_map(
            lambda g, sq: sq + jnp.square(g), updates, sum_of_squares
        )

        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        new_updates = tree_map(lambda u: -step_lr * u, new_updates)

        return new_updates, AdagradState(
            rss_state=ScaleByRssState(sum_of_squares=new_sum_of_squares),
            scale_state=ScaleByScheduleState(count=count + 1),
        )

    return GradientTransformation(init_fn, update_fn)
