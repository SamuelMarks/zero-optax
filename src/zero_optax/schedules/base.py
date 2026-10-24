"""Base learning rate schedules."""

from __future__ import annotations

from zero_optax.schedules.schedule import (
    constant_schedule,
    cosine_decay_schedule,
    exponential_decay,
    join_schedules,
    linear_schedule,
    piecewise_constant_schedule,
    piecewise_interpolate_schedule,
    polynomial_schedule,
)

__all__ = [
    "constant_schedule",
    "cosine_decay_schedule",
    "exponential_decay",
    "join_schedules",
    "linear_schedule",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polynomial_schedule",
]
