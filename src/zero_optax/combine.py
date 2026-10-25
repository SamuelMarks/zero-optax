"""Combine multiple gradient transformations sequentially."""

from __future__ import annotations

from zero_optax.base import (
    GradientTransformation,
    GradientTransformationExtraArgs,
    OptState,
    Params,
    Updates,
)


def chain(
    *args: GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
) -> GradientTransformation[tuple[OptState, ...]]:
    """Apply a sequence of chainable gradient transformations in order.

    Args:
        *args: Sequence of `GradientTransformation` or `GradientTransformationExtraArgs`
            instances to apply sequentially.

    Returns:
        A single `GradientTransformation` representing the sequential composition.
    """

    def init_fn(params: Params) -> tuple[OptState, ...]:
        """Initialize the state for every transformation in the chain.

        Args:
            params: Model parameters PyTree.

        Returns:
            Tuple of initialized optimizer states, one per transformation in `args`.
        """
        return tuple(t.init(params) for t in args)

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
    ) -> tuple[Updates, tuple[OptState, ...]]:
        """Sequentially apply each transformation's update function.

        Args:
            updates: Current gradient updates PyTree.
            state: Tuple of states corresponding to each transformation in `args`.
            params: Optional current model parameters PyTree.

        Returns:
            Tuple of `(final_updates, tuple_of_new_states)`.

        Raises:
            ValueError: If `state` is not a tuple or list of states matching `args`.
        """
        if not isinstance(state, (tuple, list)) or len(state) != len(args):
            raise ValueError(
                f"Chain state length mismatch: expected {len(args)} states, got {state!r}."
            )
        new_states: list[OptState] = []
        current_updates = updates
        for sub_state, transform in zip(state, args):
            current_updates, next_sub_state = transform.update(
                current_updates, sub_state, params
            )
            new_states.append(next_sub_state)
        return current_updates, tuple(new_states)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["chain"]
