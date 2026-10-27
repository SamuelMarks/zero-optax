"""zero_optax transforms package providing masking and combining transformations."""

from __future__ import annotations

from zero_optax.transforms import _combining, _masking
from zero_optax.transforms._combining import (
    LabelTree,
    PartitionState,
    chain,
    multi_transform,
    named_chain,
    partition,
)
from zero_optax.transforms._masking import (
    MaskedNode,
    MaskedState,
    apply_mask_tree,
    evaluate_filter_mask_tree,
    freeze,
    masked,
    resolve_mask,
)

__all__ = [
    "LabelTree",
    "MaskedNode",
    "MaskedState",
    "PartitionState",
    "_combining",
    "_masking",
    "apply_mask_tree",
    "chain",
    "evaluate_filter_mask_tree",
    "freeze",
    "masked",
    "multi_transform",
    "named_chain",
    "partition",
    "resolve_mask",
]
