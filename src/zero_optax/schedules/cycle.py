"""Cyclical learning rate schedules."""

from __future__ import annotations

from zero_optax.schedules.schedule import (
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
)

__all__ = [
    "cosine_onecycle_schedule",
    "linear_onecycle_schedule",
    "sgdr_schedule",
]
