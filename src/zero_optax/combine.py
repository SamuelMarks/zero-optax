"""Combine multiple gradient transformations sequentially (`chain`, `named_chain`, `partition`, `multi_transform`)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import NamedTuple, Union

from typing_extensions import TypeGuard
from zero_flax.nnx.filterlib import Filter, to_predicate
from zero_flax.nnx.variables import VariableBase

from zero_optax._tree import (
    _coerce_pytree_child,
    evaluate_filter_mask_tree,
)
from zero_optax._types import (
    CustomMappingPyTreeProtocol,
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
    Filter,
    type[VariableBase],
    Sequence["LabelTree"],
    Mapping[str, "LabelTree"],
    Mapping[Hashable, Union[Filter, type[VariableBase], str]],
]
PartitionLabelsArg = Union[
    LabelTree,
    Callable[[Params], LabelTree],
]


def _is_label_filter_spec(
    obj: PartitionLabelsArg | Hashable,
) -> TypeGuard[Filter | type[VariableBase]]:
    """Check whether `obj` is a `zero_flax.nnx.Filter` instance or `VariableBase` subclass.

    Args:
        obj: Candidate label or filter specification.

    Returns:
        True if `obj` is a `Filter` or `VariableBase` subclass.
    """
    return isinstance(obj, Filter) or (
        isinstance(obj, type) and issubclass(obj, VariableBase)
    )


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
        return {
            str(k): _build_label_mask(labels_v, target_label)
            for k, labels_v in labels.items()
        }
    if isinstance(labels, tuple):
        return tuple(_build_label_mask(item, target_label) for item in labels)
    if isinstance(labels, list):
        return [_build_label_mask(item, target_label) for item in labels]
    return bool(labels == target_label or str(labels) == str(target_label))


def _resolve_filter_label_map_on_params(
    filter_map: Mapping[Hashable, Filter | type[VariableBase]],
    fallback_label: Hashable | None,
    params: Params,
    prefix: tuple[str, ...] = (),
) -> MaskTree | LabelTree:
    """Assign each leaf in `params` to the first matching `Filter` key in `filter_map`.

    Args:
        filter_map: Ordered mapping from partition label to `Filter` or `VariableBase` type.
        fallback_label: Optional fallback label when no predicate matches a leaf.
        params: Target parameters PyTree.
        prefix: Current tuple of string path segments from the root.

    Returns:
        Resolved `LabelTree` matching the container structure of `params`.
    """
    if isinstance(params, MappableDataclassProtocol):
        return {
            fname: _resolve_filter_label_map_on_params(
                filter_map,
                fallback_label,
                _coerce_pytree_child(fval),
                (*prefix, fname),
            )
            for fname, fval in zip(
                params.__chex_mappable_fields__(),
                params.__chex_field_values__(),
            )
        }
    if isinstance(params, (Mapping, CustomMappingPyTreeProtocol)):
        return {
            str(k): _resolve_filter_label_map_on_params(
                filter_map,
                fallback_label,
                _coerce_pytree_child(params[k]),
                (*prefix, str(k)),
            )
            for k in params
        }
    if isinstance(params, tuple):
        return tuple(
            _resolve_filter_label_map_on_params(
                filter_map, fallback_label, item, (*prefix, str(idx))
            )
            for idx, item in enumerate(params)
        )
    if isinstance(params, list):
        return [
            _resolve_filter_label_map_on_params(
                filter_map, fallback_label, item, (*prefix, str(idx))
            )
            for idx, item in enumerate(params)
        ]
    for label_key, f_spec in filter_map.items():
        pred = to_predicate(f_spec)
        matched = (
            bool(pred(prefix, params))
            if isinstance(params, VariableBase)
            else bool(pred(prefix, None))
        )
        if matched:
            return str(label_key)
    return str(fallback_label) if fallback_label is not None else ""


def _resolve_labels_for_params(
    param_labels: PartitionLabelsArg,
    params: Params,
    transforms_keys: Sequence[Hashable],
) -> LabelTree:
    """Resolve `param_labels` (including `zero_flax.nnx.Filter` specs) against `params`.

    Args:
        param_labels: Label tree, `Filter` predicate, mapping of `{label: Filter}`,
            or callable `param_labels(params) -> LabelTree`.
        params: Target parameters PyTree.
        transforms_keys: Ordered sequence of keys from `transforms`.

    Returns:
        Resolved `LabelTree` matching `params`.
    """
    if isinstance(param_labels, (Filter, type)):
        return evaluate_filter_mask_tree(param_labels, params)
    if isinstance(param_labels, Mapping):
        filter_entries: dict[Hashable, Filter | type[VariableBase]] = {}
        fallback_key: Hashable | None = None
        for k, v in param_labels.items():
            if _is_label_filter_spec(v):
                filter_entries[k] = v
        if filter_entries and set(param_labels.keys()) <= set(transforms_keys):
            for tk in transforms_keys:
                if tk not in filter_entries:
                    fallback_key = tk
                    break
            return _resolve_filter_label_map_on_params(
                filter_entries, fallback_key, params
            )
    if callable(param_labels):
        resolved = param_labels(params)
        if isinstance(resolved, (Filter, type)):
            return evaluate_filter_mask_tree(resolved, params)
        return resolved
    return param_labels


def partition(
    transforms: Mapping[
        Hashable,
        GradientTransformation[OptState] | GradientTransformationExtraArgs[OptState],
    ],
    param_labels: PartitionLabelsArg,
    *,
    mask_compatible_extra_args: bool = False,
) -> GradientTransformationExtraArgs[PartitionState]:
    """Partition parameters by label or `zero_flax.nnx.Filter` and apply `transforms`.

    Args:
        transforms: Mapping from partition labels to gradient transformations.
        param_labels: Label prefix PyTree, `zero_flax.nnx.Filter` predicate,
            mapping `{label: Filter}`, or callable `param_labels(params) -> LabelTree`.
        mask_compatible_extra_args: Whether to also mask PyTree extra arguments.

    Returns:
        A `GradientTransformationExtraArgs` applying each partition's transformation.
    """
    transform_keys_list = list(transforms.keys())

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
            resolved_labels = _resolve_labels_for_params(
                param_labels, params, transform_keys_list
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
