"""Optimizer state path inspection, extraction, replacement, and parameter mapping utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

from typing_extensions import TypeGuard
from zero_jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

from zero_optax._tree import (
    _coerce_pytree_child,
    _rebuild_mapping,
    _rebuild_sequence,
    asarray,
    reshape,
)
from zero_optax._types import (
    CustomMappingPyTreeProtocol,
    MappableDataclassProtocol,
    OptStateProtocol,
    Params,
    PyTree,
    Tensor,
)
from zero_optax.base import OptState


@dataclass(frozen=True)
class NamedTupleKey:
    """Path entry representing a field inside a NamedTuple or OptState dataclass.

    Attributes:
        tuple_name: Class name of the NamedTuple or state container.
        name: Field name within the container.
    """

    tuple_name: str
    name: str

    def __str__(self) -> str:
        """Format the key as `'TupleName.field_name'`.

        Returns:
            Formatted string representation.
        """
        return f"{self.tuple_name}.{self.name}"


KeyEntry = Union[
    DictKey,
    FlattenedIndexKey,
    GetAttrKey,
    SequenceKey,
    NamedTupleKey,
]
KeyPath = tuple[KeyEntry, ...]
FilteringFn = Callable[[KeyPath, PyTree], bool]
ParamMapFn1 = Callable[[PyTree], PyTree]
ParamMapFn2 = Callable[[PyTree, PyTree], PyTree]
ParamMapFn3 = Callable[[PyTree, PyTree, PyTree], PyTree]
ParamMapFn4 = Callable[[PyTree, PyTree, PyTree, PyTree], PyTree]
ParamMapFn = Union[ParamMapFn1, ParamMapFn2, ParamMapFn3, ParamMapFn4]


def _is_pmap1(fn: ParamMapFn, n_extra: int) -> TypeGuard[ParamMapFn1]:
    """Narrow `fn` to 1-arg `ParamMapFn1` when `n_extra == 0`.

    Args:
        fn: Parameter mapping callable.
        n_extra: Number of extra arguments.

    Returns:
        True if `n_extra == 0`.
    """
    del fn
    return n_extra == 0


def _is_pmap2(fn: ParamMapFn, n_extra: int) -> TypeGuard[ParamMapFn2]:
    """Narrow `fn` to 2-arg `ParamMapFn2` when `n_extra == 1`.

    Args:
        fn: Parameter mapping callable.
        n_extra: Number of extra arguments.

    Returns:
        True if `n_extra == 1`.
    """
    del fn
    return n_extra == 1


def _is_pmap3(fn: ParamMapFn, n_extra: int) -> TypeGuard[ParamMapFn3]:
    """Narrow `fn` to 3-arg `ParamMapFn3` when `n_extra == 2`.

    Args:
        fn: Parameter mapping callable.
        n_extra: Number of extra arguments.

    Returns:
        True if `n_extra == 2`.
    """
    del fn
    return n_extra == 2


def _is_pmap4(fn: ParamMapFn, n_extra: int) -> TypeGuard[ParamMapFn4]:
    """Narrow `fn` to 4-arg `ParamMapFn4` when `n_extra == 3`.

    Args:
        fn: Parameter mapping callable.
        n_extra: Number of extra arguments.

    Returns:
        True if `n_extra == 3`.
    """
    del fn
    return n_extra == 3


@runtime_checkable
class Initable(Protocol):
    """Protocol for gradient transformations exposing an `init(params)` method."""

    def init(self, params: Params) -> OptState:
        """Initialize optimizer state from `params`.

        Args:
            params: Model parameters PyTree.

        Returns:
            Initialized `OptState`.
        """
        ...


def _collect_all_with_path(
    tree: PyTree,
    key: str,
    prefix: KeyPath = (),
) -> list[tuple[KeyPath, PyTree]]:
    """Recursively collect all `(path, value)` pairs matching `key` in `tree`.

    Args:
        tree: Subtree to search.
        key: Field name, dictionary key, or NamedTuple class name to match.
        prefix: Current path prefix from the root.

    Returns:
        List of matching `(KeyPath, PyTree)` pairs.
    """
    matches: list[tuple[KeyPath, PyTree]] = []
    if isinstance(tree, OptStateProtocol):
        cls_name = type(tree).__name__
        if key == cls_name:
            matches.append((prefix, tree))
        as_dict = tree._asdict()
        for fname in tree._fields:
            fval = as_dict[fname]
            child_path: KeyPath = (*prefix, NamedTupleKey(cls_name, fname))
            if fname == key:
                matches.append((child_path, fval))
            matches.extend(_collect_all_with_path(fval, key, child_path))
        return matches

    if isinstance(tree, (Mapping, CustomMappingPyTreeProtocol)):
        for k in tree:
            v = _coerce_pytree_child(tree[k])
            dict_path: KeyPath = (*prefix, DictKey(k))
            if str(k) == str(key) or k == key:
                matches.append((dict_path, v))
            matches.extend(_collect_all_with_path(v, key, dict_path))
        return matches

    if isinstance(tree, (tuple, list)):
        for idx, elem in enumerate(tree):
            seq_path: KeyPath = (*prefix, SequenceKey(idx))
            matches.extend(_collect_all_with_path(elem, key, seq_path))
        return matches

    return matches


def tree_get_all_with_path(
    tree: PyTree,
    key: str,
    filtering: FilteringFn | None = None,
) -> list[tuple[KeyPath, PyTree]]:
    """Extract all `(path, value)` pairs from `tree` matching `key`.

    Args:
        tree: PyTree or optimizer state to search in.
        key: Dictionary key, NamedTuple field name, or NamedTuple class name to search for.
        filtering: Optional predicate `filtering(path, value) -> bool` to filter matches.

    Returns:
        List of `(path, value)` pairs matching `key` and `filtering`.
    """
    found = _collect_all_with_path(tree, key, ())
    if filtering is not None:
        found = [(path, val) for path, val in found if filtering(path, val)]
    return found


def tree_get(
    tree: PyTree,
    key: str,
    default: PyTree = None,
    filtering: FilteringFn | None = None,
) -> PyTree:
    """Extract a single value from `tree` matching `key`, or return `default`.

    Args:
        tree: PyTree or optimizer state to search in.
        key: Dictionary key, NamedTuple field name, or NamedTuple class name to search for.
        default: Default value returned when no match is found.
        filtering: Optional predicate `filtering(path, value) -> bool` to filter matches.

    Returns:
        The unique matching value in `tree`, or `default` if absent.

    Raises:
        KeyError: If multiple values matching `key` and `filtering` are found in `tree`.
    """
    found = tree_get_all_with_path(tree, key, filtering=filtering)
    if len(found) > 1:
        raise KeyError(f"Found multiple values for '{key}' in {tree}.")
    if not found:
        return default
    return found[0][1]


def _replace_in_tree(
    node: PyTree,
    path: KeyPath,
    filtering: FilteringFn | None,
    replacements: Mapping[str, PyTree],
) -> PyTree:
    """Recursively replace keys/fields in `node` according to `replacements` and `filtering`.

    Args:
        node: Current PyTree node.
        path: Path to `node` from the root.
        filtering: Optional predicate `filtering(path, value) -> bool`.
        replacements: Mapping of keys/field names to replacement values.

    Returns:
        Updated PyTree node.
    """
    if isinstance(node, OptStateProtocol):
        cls_name = type(node).__name__
        if cls_name in replacements and (filtering is None or filtering(path, node)):
            return replacements[cls_name]
        as_dict = node._asdict()
        updated_vals: list[PyTree] = []
        for fname in node._fields:
            fval = as_dict[fname]
            child_path: KeyPath = (*path, NamedTupleKey(cls_name, fname))
            if fname in replacements and (
                filtering is None or filtering(child_path, fval)
            ):
                updated_vals.append(replacements[fname])
            else:
                updated_vals.append(
                    _replace_in_tree(fval, child_path, filtering, replacements)
                )
        return node._make(updated_vals)

    if isinstance(node, (Mapping, CustomMappingPyTreeProtocol)):
        m_out: dict[str, PyTree] = {}
        for k in node:
            v = _coerce_pytree_child(node[k])
            child_path = (*path, DictKey(k))
            k_str = str(k)
            if k_str in replacements and (
                filtering is None or filtering(child_path, v)
            ):
                m_out[k_str] = replacements[k_str]
            else:
                m_out[k_str] = _replace_in_tree(v, child_path, filtering, replacements)
        return _rebuild_mapping(node, m_out)

    if isinstance(node, tuple):
        return tuple(
            _replace_in_tree(elem, (*path, SequenceKey(idx)), filtering, replacements)
            for idx, elem in enumerate(node)
        )

    if isinstance(node, list):
        return [
            _replace_in_tree(elem, (*path, SequenceKey(idx)), filtering, replacements)
            for idx, elem in enumerate(node)
        ]

    return node


def tree_set(
    tree: PyTree,
    filtering: FilteringFn | None = None,
    /,
    **kwargs: PyTree,
) -> PyTree:
    """Create a copy of `tree` with values matching `**kwargs` replaced.

    Args:
        tree: PyTree whose matching keys/fields are replaced.
        filtering: Optional predicate `filtering(path, value) -> bool` restricting replacements.
        **kwargs: Key/field names and their replacement values.

    Returns:
        Updated PyTree with identical structure and replaced values.

    Raises:
        KeyError: If any key in `kwargs` is not found in `tree` (or does not pass `filtering`).
    """
    for key in kwargs:
        found = tree_get_all_with_path(tree, key, filtering=filtering)
        if not found:
            if filtering is not None:
                raise KeyError(
                    f"Found no values matching '{key}' given the filtering operation in {tree}"
                )
            raise KeyError(f"Found no values matching '{key}' in {tree}")
    return _replace_in_tree(tree, (), filtering, kwargs)


def _call_param_fn(
    fn: ParamMapFn,
    primary: PyTree,
    rest: Sequence[PyTree],
) -> PyTree:
    """Invoke `fn` with `primary` and `rest` using strict arity TypeGuards.

    Args:
        fn: Parameter mapping callable taking 1 to 4 PyTree arguments.
        primary: Primary parameter leaf/subtree.
        rest: Additional parameter leaves/subtrees.

    Returns:
        Result of `fn(primary, *rest)`.

    Raises:
        ValueError: If `len(rest) > 3`.
    """
    n_extra = len(rest)
    if _is_pmap1(fn, n_extra):
        return fn(primary)
    if _is_pmap2(fn, n_extra):
        return fn(primary, rest[0])
    if _is_pmap3(fn, n_extra):
        return fn(primary, rest[0], rest[1])
    if _is_pmap4(fn, n_extra):
        return fn(primary, rest[0], rest[1], rest[2])
    raise ValueError(f"tree_map_params supports up to 3 extra trees, got {n_extra}.")


def _map_with_is_leaf(
    fn: ParamMapFn,
    primary: PyTree,
    rest: Sequence[PyTree],
    is_leaf: Callable[[PyTree], bool] | None,
) -> PyTree:
    """Map `fn` across `primary` and `rest` honoring an optional `is_leaf` predicate.

    Args:
        fn: Callable applied to each parameter leaf (or `is_leaf` subtree).
        primary: Primary parameter tree inside the optimizer state.
        rest: Additional parameter-shaped trees passed to `fn`.
        is_leaf: Optional predicate stopping traversal at matching nodes.

    Returns:
        Mapped PyTree.
    """
    if is_leaf is not None and is_leaf(primary):
        return _call_param_fn(fn, primary, rest)
    if isinstance(primary, MappableDataclassProtocol):
        dc_out: dict[str, PyTree] = {}
        fields = primary.__chex_mappable_fields__()
        vals = [_coerce_pytree_child(v) for v in primary.__chex_field_values__()]
        for idx, (fname, fval) in enumerate(zip(fields, vals)):
            sub_rest_dc: list[PyTree] = [
                _coerce_pytree_child(r.__chex_field_values__()[idx])
                if isinstance(r, MappableDataclassProtocol)
                else r
                for r in rest
            ]
            dc_out[fname] = _map_with_is_leaf(fn, fval, sub_rest_dc, is_leaf)
        cls_dc = type(primary)
        return cls_dc(**dc_out)
    if isinstance(primary, (Mapping, CustomMappingPyTreeProtocol)):
        m_res: dict[str, PyTree] = {}
        for k in primary:
            sub_rest_m: list[PyTree] = [
                _coerce_pytree_child(r[k])
                if isinstance(r, (Mapping, CustomMappingPyTreeProtocol))
                else r
                for r in rest
            ]
            m_res[str(k)] = _map_with_is_leaf(
                fn, _coerce_pytree_child(primary[k]), sub_rest_m, is_leaf
            )
        return _rebuild_mapping(primary, m_res)
    if isinstance(primary, (tuple, list)):
        seq_items: list[PyTree] = []
        for idx, elem in enumerate(primary):
            sub_rest_seq: list[PyTree] = [
                r[idx] if isinstance(r, (tuple, list)) else r for r in rest
            ]
            seq_items.append(_map_with_is_leaf(fn, elem, sub_rest_seq, is_leaf))
        return _rebuild_sequence(primary, seq_items)
    return _call_param_fn(fn, primary, rest)


_SENTINEL_SHAPE: tuple[int, ...] = (3, 1, 7)


def _is_sentinel_subtree(node: PyTree) -> bool:
    """Check whether `node` is a parameter-shaped subtree produced from the sentinel tensor.

    Args:
        node: Candidate node inside `state_with_sentinel`.

    Returns:
        True if `node` is a Tensor with shape `_SENTINEL_SHAPE`.
    """
    return isinstance(node, Tensor) and node.shape == _SENTINEL_SHAPE


def _map_state_against_sentinel(
    sentinel_node: PyTree,
    state_node: PyTree,
    f: ParamMapFn,
    rest: Sequence[PyTree],
    transform_non_params: Callable[[PyTree], PyTree] | None,
    is_leaf: Callable[[PyTree], bool] | None,
) -> PyTree:
    """Recursively traverse `sentinel_node` and `state_node` to map parameter trees with `f`.

    Args:
        sentinel_node: Node from `init(sentinel_tensor)`.
        state_node: Corresponding node from actual `state`.
        f: Function applied to parameter-shaped trees.
        rest: Additional parameter-shaped arguments passed to `f`.
        transform_non_params: Optional function applied to non-parameter leaves.
        is_leaf: Optional leaf predicate passed to parameter tree mapping.

    Returns:
        Mapped optimizer state node.
    """
    if _is_sentinel_subtree(sentinel_node):
        return _map_with_is_leaf(f, state_node, rest, is_leaf)

    if isinstance(sentinel_node, OptStateProtocol) and isinstance(
        state_node, OptStateProtocol
    ):
        sent_dict = sentinel_node._asdict()
        state_dict = state_node._asdict()
        updated_vals = [
            _map_state_against_sentinel(
                sent_dict[fname],
                state_dict[fname],
                f,
                rest,
                transform_non_params,
                is_leaf,
            )
            for fname in state_node._fields
        ]
        return state_node._make(updated_vals)

    if isinstance(sentinel_node, (Mapping, CustomMappingPyTreeProtocol)) and isinstance(
        state_node, (Mapping, CustomMappingPyTreeProtocol)
    ):
        sent_keys = set(sentinel_node.keys())
        m_updated: dict[str, PyTree] = {}
        for k in state_node:
            v_st = _coerce_pytree_child(state_node[k])
            if k in sent_keys:
                m_updated[str(k)] = _map_state_against_sentinel(
                    _coerce_pytree_child(sentinel_node[k]),
                    v_st,
                    f,
                    rest,
                    transform_non_params,
                    is_leaf,
                )
            else:
                m_updated[str(k)] = (
                    transform_non_params(v_st)
                    if transform_non_params is not None
                    else v_st
                )
        return _rebuild_mapping(state_node, m_updated)

    if isinstance(sentinel_node, tuple) and isinstance(state_node, tuple):
        return tuple(
            _map_state_against_sentinel(
                s_elem, st_elem, f, rest, transform_non_params, is_leaf
            )
            for s_elem, st_elem in zip(sentinel_node, state_node)
        )

    if isinstance(sentinel_node, list) and isinstance(state_node, list):
        return [
            _map_state_against_sentinel(
                s_elem, st_elem, f, rest, transform_non_params, is_leaf
            )
            for s_elem, st_elem in zip(sentinel_node, state_node)
        ]

    if transform_non_params is not None:
        return transform_non_params(state_node)
    return state_node


def tree_map_params(
    initable: Initable | Callable[[Params], OptState],
    f: ParamMapFn,
    state: OptState,
    /,
    *rest: PyTree,
    transform_non_params: Callable[[PyTree], PyTree] | None = None,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> OptState:
    """Apply `f` over all parameter-shaped subtrees inside optimizer state `state`.

    Args:
        initable: Gradient transformation (with `.init`) or init callable `(params) -> state`.
        f: Callable applied to each parameter-shaped tree leaf inside `state`.
        state: Optimizer state to map over.
        *rest: Additional parameter-shaped PyTrees passed alongside each parameter leaf to `f`.
        transform_non_params: Optional callable applied to non-parameter fields in `state`.
        is_leaf: Optional predicate stopping parameter tree traversal at matching subtrees.

    Returns:
        Updated optimizer state with `f` applied to all parameter subtrees.
    """
    sentinel = reshape(full_like_sentinel(), _SENTINEL_SHAPE)
    if isinstance(initable, Initable):
        sentinel_state = initable.init(sentinel)
    else:
        sentinel_state = initable(sentinel)
    return _map_state_against_sentinel(
        sentinel_state,
        state,
        f,
        rest,
        transform_non_params,
        is_leaf,
    )


def full_like_sentinel() -> Tensor:
    """Create a 1D sentinel zero tensor of length 21 for `_SENTINEL_SHAPE = (3, 1, 7)`.

    Returns:
        1D zero Tensor of length 21.
    """
    return asarray([0.0] * 21)


__all__ = [
    "FilteringFn",
    "Initable",
    "KeyEntry",
    "KeyPath",
    "NamedTupleKey",
    "ParamMapFn",
    "tree_get",
    "tree_get_all_with_path",
    "tree_map_params",
    "tree_set",
]
