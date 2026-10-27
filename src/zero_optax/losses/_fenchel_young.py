"""Fenchel-Young loss construction utilities."""

from __future__ import annotations

from zero_optax.losses.fenchel_young import (
    FenchelYoungLossFn,
    MaxFun,
    make_fenchel_young_loss,
)

__all__ = [
    "FenchelYoungLossFn",
    "MaxFun",
    "make_fenchel_young_loss",
]
