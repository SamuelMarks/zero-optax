"""Combine multiple gradient transformations."""

from typing import Optional, Tuple
from zero_optax.base import GradientTransformation, OptState, Params, Updates


def chain(*args: GradientTransformation) -> GradientTransformation:
    """Applies a list of chainable update transformations.

    Args:
        *args: a sequence of chainable (init, update) tuples.

    Returns:
        A single `GradientTransformation`.
    """

    def init_fn(params: Params) -> OptState:
        return tuple(t.init(params) for t in args)

    def update_fn(
        updates: Updates, state: OptState, params: Optional[Params] = None
    ) -> Tuple[Updates, OptState]:
        new_state = []
        for s, t in zip(state, args):
            updates, new_s = t.update(updates, s, params)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return GradientTransformation(init_fn, update_fn)
