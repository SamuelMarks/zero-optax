"""Contrastive and metric learning loss functions."""

from __future__ import annotations

from zero_optax.losses.metric import ntxent, triplet_margin_loss

__all__ = [
    "ntxent",
    "triplet_margin_loss",
]
