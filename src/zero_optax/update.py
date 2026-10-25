"""Parameter update and norm calculation utilities."""

from __future__ import annotations

from collections.abc import Mapping

import zero_chex
from zero_chex.types import PyTree as ChexPyTree

from zero_optax._tree import (
    _coerce_pytree_child,
    asarray,
    tree_l2_norm,
    tree_map,
    where,
)
from zero_optax._types import (
    CustomMappingPyTreeProtocol,
    MappableDataclassProtocol,
    Params,
    PyTree,
    Tensor,
    TensorLeaf,
    Updates,
    VariableLeafProtocol,
)


def _to_chex_tree(tree: PyTree) -> ChexPyTree:
    """Normalize a `zero_optax` `PyTree` into a `zero_chex` `PyTree` for structural assertions.

    Args:
        tree: Input `zero_optax` `PyTree`.

    Returns:
        Equivalent `zero_chex` `PyTree`.
    """
    if isinstance(tree, MappableDataclassProtocol):
        return {
            k: _to_chex_tree(_coerce_pytree_child(v))
            for k, v in zip(
                tree.__chex_mappable_fields__(), tree.__chex_field_values__()
            )
        }
    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        return {str(k): _to_chex_tree(_coerce_pytree_child(tree[k])) for k in tree}
    if isinstance(tree, tuple):
        return tuple(_to_chex_tree(item) for item in tree)
    if isinstance(tree, list):
        return [_to_chex_tree(item) for item in tree]
    if isinstance(tree, VariableLeafProtocol):
        return asarray(tree.value)
    if isinstance(tree, (Tensor, float, int, bool, type(None))):
        return tree
    return asarray(tree)


def apply_updates(params: Params, updates: Updates) -> Params:
    """Apply gradient updates to parameters by element-wise addition.

    Args:
        params: Current model parameters PyTree.
        updates: Transformed gradient updates PyTree matching `params`.

    Returns:
        Updated parameters PyTree `params + updates`.
    """
    zero_chex.assert_trees_all_equal_structs(
        _to_chex_tree(params), _to_chex_tree(updates)
    )

    def _add_leaf(p: TensorLeaf, u: TensorLeaf) -> TensorLeaf:
        """Add update leaf `u` to parameter leaf `p`.

        Args:
            p: Parameter leaf.
            u: Update leaf.

        Returns:
            Sum `p + u` as a Tensor, or `p` if either is None.
        """
        if p is None or u is None:
            return p
        return asarray(p) + asarray(u)

    return tree_map(_add_leaf, params, updates)


def incremental_update(
    new_tensors: Params,
    old_tensors: Params,
    step_size: float | Tensor,
) -> Params:
    """Incrementally update parameters via Polyak / exponential moving average.

    Computes `step_size * new_tensors + (1.0 - step_size) * old_tensors`.

    Args:
        new_tensors: New target parameter PyTree.
        old_tensors: Old slow-moving parameter PyTree.
        step_size: Interpolation factor in `[0, 1]`.

    Returns:
        Interpolated parameter PyTree.
    """
    zero_chex.assert_trees_all_equal_structs(
        _to_chex_tree(new_tensors), _to_chex_tree(old_tensors)
    )

    def _interp_leaf(new_v: TensorLeaf, old_v: TensorLeaf) -> TensorLeaf:
        """Interpolate between `new_v` and `old_v`.

        Args:
            new_v: New tensor leaf.
            old_v: Old tensor leaf.

        Returns:
            Interpolated Tensor leaf.
        """
        if new_v is None or old_v is None:
            return new_v
        return asarray(step_size) * asarray(new_v) + (
            1.0 - asarray(step_size)
        ) * asarray(old_v)

    return tree_map(_interp_leaf, new_tensors, old_tensors)


def periodic_update(
    new_tensors: Params,
    old_tensors: Params,
    steps: int | Tensor,
    update_period: int,
) -> Params:
    """Periodically replace `old_tensors` with `new_tensors` every `update_period` steps.

    Args:
        new_tensors: Latest parameter PyTree.
        old_tensors: Target parameter PyTree updated periodically.
        steps: Current integer step counter.
        update_period: Period (in steps) at which to synchronize parameters.

    Returns:
        `new_tensors` when `steps % update_period == 0`, otherwise `old_tensors`.
    """
    zero_chex.assert_trees_all_equal_structs(
        _to_chex_tree(new_tensors), _to_chex_tree(old_tensors)
    )
    should_update = (asarray(steps) % update_period) == 0

    def _select_leaf(new_v: TensorLeaf, old_v: TensorLeaf) -> TensorLeaf:
        """Select `new_v` or `old_v` based on `should_update`.

        Args:
            new_v: New tensor leaf.
            old_v: Old tensor leaf.

        Returns:
            Selected Tensor leaf.
        """
        if new_v is None or old_v is None:
            return new_v
        return where(should_update, asarray(new_v), asarray(old_v))

    return tree_map(_select_leaf, new_tensors, old_tensors)


def global_norm(updates: PyTree) -> Tensor:
    """Compute the global L2 norm across all leaves in a PyTree.

    Args:
        updates: Input PyTree of tensors.

    Returns:
        Scalar Tensor holding the global L2 norm.
    """
    return tree_l2_norm(updates, squared=False)


__all__ = [
    "apply_updates",
    "global_norm",
    "incremental_update",
    "periodic_update",
]
