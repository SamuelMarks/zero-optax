"""Combine multiple gradient transformations sequentially (`chain`, `named_chain`, `partition`, `multi_transform`)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import NamedTuple, Union

from zero_optax._tree import (
    _coerce_pytree_child,
)
from zero_optax._types import (
    MappableDataclassProtocol,
    MaskTree,
    Params,
    PyTree,
    Updates,
)
from zero_optax.base import (
    ExtraArgValue,
    GradientTransformation,
    GradientTransformationExtraArgs,
    OptState,
)
from zero_optax.wrappers import MaskedState, masked


class PartitionState(NamedTuple):
    """State for `partition` / `multi_transform` mapping each partition label to its `MaskedState`.

    Attributes:
        inner_states: Dictionary mapping partition labels to `MaskedState` instances.
    """

    inner_states: dict[str, MaskedState]


def chain(
    *args: GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
) -> GradientTransformationExtraArgs[tuple[OptState, ...]]:
    """Apply a sequence of chainable gradient transformations in order.

    Args:
        *args: Sequence of `GradientTransformation` or `GradientTransformationExtraArgs`
            instances to apply sequentially.

    Returns:
        A single `GradientTransformationExtraArgs` representing the sequential composition.
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
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, tuple[OptState, ...]]:
        """Sequentially apply each transformation's update function.

        Args:
            updates: Current gradient updates PyTree.
            state: Tuple of states corresponding to each transformation in `args`.
            params: Optional current model parameters PyTree.
            **extra_args: Extra keyword arguments forwarded to `GradientTransformationExtraArgs` steps.

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
            if isinstance(transform, GradientTransformationExtraArgs):
                current_updates, next_sub_state = transform.update(
                    current_updates, sub_state, params, **extra_args
                )
            else:
                current_updates, next_sub_state = transform.update(
                    current_updates, sub_state, params
                )
            new_states.append(next_sub_state)
        return current_updates, tuple(new_states)

    return GradientTransformationExtraArgs(init_fn, update_fn)


def named_chain(
    *args: tuple[
        str,
        GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
    ],
) -> GradientTransformationExtraArgs[dict[str, OptState]]:
    """Apply a list of `(name, transformation)` pairs sequentially with a dict-keyed state.

    Args:
        *args: Sequence of `(name, transform)` tuples with unique string names.

    Returns:
        A `GradientTransformationExtraArgs` storing sub-states in a dict keyed by `name`.

    Raises:
        ValueError: If duplicate transformation names are present in `args`.
    """
    names = [name for name, _ in args]
    if len(names) != len(set(names)):
        raise ValueError(
            f"Named transformations must have unique names, but got {names}."
        )

    def init_fn(params: Params) -> dict[str, OptState]:
        """Initialize the named dictionary state for all `(name, tx)` pairs.

        Args:
            params: Model parameters PyTree.

        Returns:
            Dictionary mapping each `name` to `tx.init(params)`.
        """
        return {name: tx.init(params) for name, tx in args}

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, dict[str, OptState]]:
        """Apply each `(name, tx)` in order and return `(updates, new_named_state)`.

        Args:
            updates: Gradient updates PyTree.
            state: Dictionary mapping transformation names to sub-states.
            params: Optional model parameters PyTree.
            **extra_args: Extra keyword arguments forwarded to `GradientTransformationExtraArgs` steps.

        Returns:
            Tuple of `(transformed_updates, new_named_state_dict)`.

        Raises:
            TypeError: If `state` is not a `Mapping`.
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                f"Expected mapping state in named_chain, got {type(state)!r}."
            )
        new_state: dict[str, OptState] = {}
        current_updates = updates
        for name, tx in args:
            sub_st = state[name]
            if isinstance(tx, GradientTransformationExtraArgs):
                current_updates, next_st = tx.update(
                    current_updates, sub_st, params, **extra_args
                )
            else:
                current_updates, next_st = tx.update(current_updates, sub_st, params)
            new_state[name] = next_st
        return current_updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


LabelTree = Union[
    PyTree,
    str,
    Sequence["LabelTree"],
    Mapping[str, "LabelTree"],
]


def _build_label_mask(
    labels: LabelTree,
    target_label: Hashable,
) -> MaskTree:
    """Convert a label prefix PyTree into a boolean `MaskTree` for `target_label`.

    Args:
        labels: Label prefix PyTree (whose leaves are label strings/values).
        target_label: Partition key to match (`True` where `leaf == target_label`).

    Returns:
        Boolean `MaskTree` matching the prefix structure of `labels`.
    """
    if isinstance(labels, str):
        return bool(labels == target_label or str(labels) == str(target_label))
    if isinstance(labels, MappableDataclassProtocol):
        return {
            fname: _build_label_mask(_coerce_pytree_child(fval), target_label)
            for fname, fval in zip(
                labels.__chex_mappable_fields__(),
                labels.__chex_field_values__(),
            )
        }
    if isinstance(labels, Mapping):
        return {str(k): _build_label_mask(labels[k], target_label) for k in labels}
    if isinstance(labels, tuple):
        return tuple(_build_label_mask(item, target_label) for item in labels)
    if isinstance(labels, list):
        return [_build_label_mask(item, target_label) for item in labels]
    return bool(labels == target_label or str(labels) == str(target_label))


def partition(
    transforms: Mapping[
        Hashable,
        GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
    ],
    param_labels: LabelTree | Callable[[Params], LabelTree],
    *,
    mask_compatible_extra_args: bool = False,
) -> GradientTransformationExtraArgs[PartitionState]:
    """Partition parameters by label and apply the corresponding transformation from `transforms`.

    Args:
        transforms: Mapping from partition labels to gradient transformations.
        param_labels: Label prefix PyTree (or callable `param_labels(params) -> LabelTree`).
        mask_compatible_extra_args: Whether to also mask PyTree extra arguments.

    Returns:
        A `GradientTransformationExtraArgs` applying each partition's transformation.
    """

    def _make_mask_for_group(group: Hashable) -> Callable[[Params], MaskTree]:
        """Create a mask callable selecting only parameters labeled with `group`.

        Args:
            group: Partition label key.

        Returns:
            Mask callable `(params) -> MaskTree`.
        """

        def _mask_fn(params: Params) -> MaskTree:
            """Resolve `param_labels` on `params` and compare against `group`.

            Args:
                params: Model parameters PyTree.

            Returns:
                Boolean `MaskTree` for `group`.
            """
            resolved_labels = (
                param_labels(params) if callable(param_labels) else param_labels
            )
            return _build_label_mask(resolved_labels, group)

        return _mask_fn

    wrapped_pairs: list[tuple[str, GradientTransformationExtraArgs[MaskedState]]] = [
        (
            str(group),
            masked(
                tx,
                _make_mask_for_group(group),
                mask_compatible_extra_args=mask_compatible_extra_args,
            ),
        )
        for group, tx in transforms.items()
    ]

    def init_fn(params: Params) -> PartitionState:
        """Initialize `PartitionState` across all partition groups.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `PartitionState`.
        """
        return PartitionState(
            inner_states={
                group_key: masked_tx.init(params)
                for group_key, masked_tx in wrapped_pairs
            }
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        **extra_args: ExtraArgValue,
    ) -> tuple[Updates, PartitionState]:
        """Apply each partition's masked transformation and return `(updates, new_state)`.

        Args:
            updates: Gradient updates PyTree.
            state: Current `PartitionState`.
            params: Optional model parameters PyTree.
            **extra_args: Extra keyword arguments forwarded to partition transformations.

        Returns:
            Tuple of `(partitioned_updates, PartitionState(new_inner_states))`.

        Raises:
            TypeError: If `state` is not a `PartitionState`.
        """
        if not isinstance(state, PartitionState):
            raise TypeError(
                f"Expected PartitionState in partition, got {type(state)!r}."
            )
        new_inner: dict[str, MaskedState] = {}
        current_updates = updates
        for group_key, masked_tx in wrapped_pairs:
            current_updates, next_masked_st = masked_tx.update(
                current_updates,
                state.inner_states[group_key],
                params,
                **extra_args,
            )
            new_inner[group_key] = next_masked_st
        return current_updates, PartitionState(inner_states=new_inner)

    return GradientTransformationExtraArgs(init_fn, update_fn)


multi_transform = partition

__all__ = [
    "LabelTree",
    "PartitionState",
    "chain",
    "multi_transform",
    "named_chain",
    "partition",
]
