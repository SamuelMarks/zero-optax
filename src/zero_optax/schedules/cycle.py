"""Module docstring."""

"""Cycle schedules."""
from zero_optax.schedules.schedule import (
    cosine_onecycle_schedule,
    linear_onecycle_schedule,
    sgdr_schedule,
)

__all__ = ["cosine_onecycle_schedule", "linear_onecycle_schedule", "sgdr_schedule"]
