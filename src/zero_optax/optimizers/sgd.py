"""Stochastic Gradient Descent."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import zero_jax.numpy as jnp
from zero_jax.tree_util import tree_map

from zero_optax.base import GradientTransformation, OptState, Params, Updates


class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""

    trace: Updates


class EmptyState(NamedTuple):
    """An empty state for stateless transformations."""


class ScaleByScheduleState(NamedTuple):
    """Maintains count for scale schedules."""

    count: jnp.ndarray


class SgdState(NamedTuple):
    """State for the SGD optimizer."""

    trace_state: Union[TraceState, EmptyState]
    scale_state: ScaleByScheduleState


def sgd(
    learning_rate: Union[float, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]],
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> GradientTransformation:
    """A canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for
    momentum, and Nesterov acceleration, as these are standard practice when
    using stochastic gradient descent to train deep neural networks.


    The canonical stochastic gradient descent returns an update
    :math:`u_t` of the form

    .. math::
      u_t \\leftarrow -\\alpha_t g_t,

    where :math:`g_t` is the gradient of the objective (potentially preprocessed
    by other transformations) and :math:`\\alpha_t` is the ``learning_rate`` at
    time :math:`t` (constant or selected by an :class:`optax.Schedule`).

    Stochastic gradient descent with momentum takes two possible forms.

    .. math::

      \\begin{align*}
        m_t &\\leftarrow g_t + \\mu m_{t-1} \\\\
        u_t &\\leftarrow \\begin{cases}
          -\\alpha_t m_t & \\text{ if } \\texttt{nesterov = False} \\\\
          -\\alpha_t (g_t + \\mu m_t) & \\text{ if } \\texttt{nesterov = True}
          \\end{cases} \\\\
        S_t &\\leftarrow m_t,
      \\end{align*}

    where :math:`\\mu` is the ``momentum`` parameter and :math:`S_t` is the state
    of the optimizer.

    Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
      momentum: Decay rate used by the momentum term, when it is set to ``None``,
        then momentum is not used at all.
      nesterov: Whether Nesterov momentum is used.
      accumulator_dtype: Optional ``dtype`` to be used for the accumulator; if
        ``None`` then the ``dtype`` is inferred from ``params`` and ``updates``.

    Returns:
      The corresponding :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Params) -> SgdState:
        trace_state: Union[TraceState, EmptyState]
        if momentum is not None:
            trace_state = TraceState(
                trace=tree_map(
                    lambda t: jnp.zeros_like(
                        t,
                        dtype=accumulator_dtype
                        if accumulator_dtype is not None
                        else t.dtype,
                    ),
                    params,
                )
            )
        else:
            trace_state = EmptyState()

        return SgdState(
            trace_state=trace_state,
            scale_state=ScaleByScheduleState(count=jnp.array(0)),
        )

    def update_fn(
        updates: Updates, state: SgdState, params: Optional[Params] = None
    ) -> Tuple[Updates, SgdState]:
        # Momentum step
        trace_state: Union[TraceState, EmptyState]
        if momentum is not None:
            trace_state = state.trace_state
            assert isinstance(trace_state, TraceState)

            def update_momentum(g, t):
                return g + momentum * t

            new_trace = tree_map(update_momentum, updates, trace_state.trace)

            if nesterov:
                updates = tree_map(update_momentum, updates, new_trace)
            else:
                updates = new_trace

            new_trace_state: Union[TraceState, EmptyState] = TraceState(trace=new_trace)
        else:
            new_trace_state = EmptyState()

        # Learning rate step
        count = state.scale_state.count
        if callable(learning_rate):
            step_lr = learning_rate(count)
        else:
            step_lr = learning_rate

        updates = tree_map(lambda g: -step_lr * g, updates)

        new_scale_state = ScaleByScheduleState(count=count + 1)

        return updates, SgdState(
            trace_state=new_trace_state, scale_state=new_scale_state
        )

    return GradientTransformation(init_fn, update_fn)
