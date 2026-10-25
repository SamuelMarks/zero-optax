"""Random PyTree utilities (`tree_split_key_like`, `tree_random_like`, `tree_unwrap_random_key_data`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from zero_optax._tree import (
    _coerce_pytree_child,
    _rebuild_mapping,
    _rebuild_sequence,
    asarray,
    reshape,
    tree_leaves,
    tree_map,
)
from zero_optax._types import (
    ArrayLike,
    CustomMappingPyTreeProtocol,
    DTypeLike,
    MappableDataclassProtocol,
    PyTree,
    Tensor,
    TensorLeaf,
)
from zero_optax.tree_utils._tree_math import _flatten_tensor_floats

SamplerFn = Callable[[Tensor, tuple[int, ...], DTypeLike], ArrayLike]


def _extract_seed_int(rng_key: ArrayLike | int) -> int:
    """Extract an integer seed from a PRNGKey tensor, array-like, or integer.

    Args:
        rng_key: PRNG key array or integer seed.

    Returns:
        Deterministic non-negative 32-bit integer seed.
    """
    if isinstance(rng_key, int):
        return rng_key & 0xFFFFFFFF
    t = asarray(rng_key)
    vals = _flatten_tensor_floats(t)
    acc = 0
    for idx, v in enumerate(vals):
        acc = (acc ^ (int(v) * (2654435761 + idx))) & 0xFFFFFFFF
    return acc


def _split_seed_to_keys(base_seed: int, num_keys: int) -> list[Tensor]:
    """Deterministically split `base_seed` into `num_keys` 2-element integer key tensors.

    Args:
        base_seed: Integer base seed.
        num_keys: Number of sub-keys to generate.

    Returns:
        List of `num_keys` `Tensor([k0, k1], dtype='int32')` keys.
    """
    keys: list[Tensor] = []
    for idx in range(num_keys):
        h0 = ((base_seed + idx * 0x9E3779B9) * 0x85EBCA6B) & 0x7FFFFFFF
        h1 = ((h0 ^ (idx + 1) * 0xC2B2AE35) * 0x27D4EB2D) & 0x7FFFFFFF
        keys.append(asarray([h0, h1], dtype="int32"))
    return keys


def _default_normal_sampler(
    key: Tensor,
    shape: tuple[int, ...],
    dtype: DTypeLike = None,
) -> Tensor:
    """Deterministic Box-Muller standard normal sampler matching `(key, shape, dtype)`.

    Args:
        key: 2-element PRNG key Tensor.
        shape: Output tensor shape.
        dtype: Optional target dtype.

    Returns:
        Standard normal Tensor of shape `shape`.
    """
    seed = _extract_seed_int(key)
    total = 1
    for dim in shape:
        total *= dim
    samples: list[float] = []
    state = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
    for idx in range((total + 1) // 2):
        state = (1664525 * state + 1013904223 + idx) & 0xFFFFFFFF
        u1 = max((state + 1.0) / 4294967297.0, 1e-12)
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u2 = (state + 1.0) / 4294967297.0
        mag = math.sqrt(-2.0 * math.log(u1))
        z0 = mag * math.cos(2.0 * math.pi * u2)
        z1 = mag * math.sin(2.0 * math.pi * u2)
        samples.append(z0)
        if len(samples) < total:
            samples.append(z1)
    if total == 0:
        return reshape(asarray([], dtype=dtype), shape)
    if shape == ():
        return asarray(samples[0], dtype=dtype)
    return reshape(asarray(samples[:total], dtype=dtype), shape)


def _unflatten_like(
    template: PyTree,
    leaves: Sequence[TensorLeaf],
    idx: int = 0,
) -> tuple[PyTree, int]:
    """Reconstruct a PyTree with the structure of `template` from flat `leaves`.

    Args:
        template: Template PyTree defining container structure.
        leaves: Sequence of replacement leaves.
        idx: Current index in `leaves`.

    Returns:
        Tuple `(reconstructed_tree, next_idx)`.
    """
    if isinstance(template, MappableDataclassProtocol):
        dc_updates: dict[str, PyTree] = {}
        for fname, fval in zip(
            template.__chex_mappable_fields__(),
            template.__chex_field_values__(),
        ):
            dc_updates[fname], idx = _unflatten_like(
                _coerce_pytree_child(fval), leaves, idx
            )
        cls_dc = type(template)
        return cls_dc(**dc_updates), idx
    if isinstance(template, (Mapping, CustomMappingPyTreeProtocol)):
        m_items: dict[str, PyTree] = {}
        for k in template:
            m_items[str(k)], idx = _unflatten_like(
                _coerce_pytree_child(template[k]), leaves, idx
            )
        return _rebuild_mapping(template, m_items), idx
    if isinstance(template, (tuple, list)):
        seq_vals: list[PyTree] = []
        for item in template:
            val, idx = _unflatten_like(item, leaves, idx)
            seq_vals.append(val)
        return _rebuild_sequence(template, seq_vals), idx
    return leaves[idx], idx + 1


def tree_split_key_like(
    rng_key: ArrayLike | int,
    target_tree: PyTree,
) -> PyTree:
    """Split a PRNG key to match the PyTree structure of `target_tree`.

    Args:
        rng_key: Base PRNG key array or integer seed.
        target_tree: Target PyTree whose leaf structure is matched.

    Returns:
        PyTree of split PRNG keys matching the structure of `target_tree`.
    """
    flat_leaves = tree_leaves(target_tree)
    base_seed = _extract_seed_int(rng_key)
    subkeys = _split_seed_to_keys(base_seed, len(flat_leaves))
    key_tree, _ = _unflatten_like(target_tree, subkeys, 0)
    return key_tree


def tree_random_like(
    rng_key: ArrayLike | int,
    target_tree: PyTree,
    sampler: SamplerFn = _default_normal_sampler,
    dtype: DTypeLike = None,
) -> PyTree:
    """Create a PyTree of random tensors matching the structure and shapes of `target_tree`.

    Args:
        rng_key: Base PRNG key array or integer seed.
        target_tree: Target PyTree whose structure and leaf shapes are matched.
        sampler: Sampling callable `(key, shape, dtype) -> Tensor` (defaults to normal).
        dtype: Optional target dtype for sampled leaves (defaults to each leaf's dtype).

    Returns:
        Random PyTree matching `target_tree`.
    """
    keys_tree = tree_split_key_like(rng_key, target_tree)

    def _sample_leaf(leaf: TensorLeaf, key: TensorLeaf) -> TensorLeaf:
        """Sample a single random leaf using `sampler`.

        Args:
            leaf: Template leaf from `target_tree`.
            key: Split PRNG key leaf.

        Returns:
            Sampled Tensor leaf.
        """
        if leaf is None or key is None:
            return None
        leaf_t = asarray(leaf)
        key_t = asarray(key)
        target_dtype: DTypeLike = dtype if dtype is not None else leaf_t.dtype
        return asarray(sampler(key_t, leaf_t.shape, target_dtype), dtype=target_dtype)

    return tree_map(_sample_leaf, target_tree, keys_tree)


def tree_unwrap_random_key_data(input_tree: PyTree) -> PyTree:
    """Unwrap random key objects in `input_tree` into underlying tensor data.

    Args:
        input_tree: PyTree containing arrays and/or PRNG keys.

    Returns:
        PyTree with PRNG keys unwrapped into raw tensor data.
    """

    def _unwrap_leaf(leaf: TensorLeaf) -> TensorLeaf:
        """Convert leaf to raw Tensor.

        Args:
            leaf: Input leaf.

        Returns:
            Unwrapped Tensor leaf.
        """
        if leaf is None:
            return None
        return asarray(leaf)

    return tree_map(_unwrap_leaf, input_tree)


__all__ = [
    "SamplerFn",
    "tree_random_like",
    "tree_split_key_like",
    "tree_unwrap_random_key_data",
]
